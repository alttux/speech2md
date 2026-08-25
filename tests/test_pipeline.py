import pytest

from speech2md.core import pipeline
from speech2md.core.models import FormatMode, Segment, Settings, TranscriptionResult
from speech2md.core.postprocess.llm_client import LLMClient


class FakeTranscriber:
    def __init__(self, **kwargs: object) -> None:
        pass

    def transcribe(self, audio_path, language="", on_progress=None):  # type: ignore[no-untyped-def]
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        return TranscriptionResult(
            segments=[Segment(start=0.0, end=2.0, text="привет мир")],
            full_text="Первое предложение тут. Второе предложение здесь.",
            language="ru",
            duration=2.0,
        )


class FakeLLM(LLMClient):
    def __init__(self, response: str = "PROCESSED") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class BrokenLLM(LLMClient):
    def complete(self, prompt: str) -> str:
        raise RuntimeError("llm unreachable")


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "lecture.mp3"
    path.write_bytes(b"not really audio")
    return path


@pytest.fixture
def stub_pipeline(tmp_path, monkeypatch):
    settings = Settings(output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(pipeline, "load_config", lambda: settings)
    monkeypatch.setattr(pipeline, "WhisperTranscriber", FakeTranscriber)

    def use_llm(llm: LLMClient) -> None:
        monkeypatch.setattr(pipeline, "create_client", lambda **kwargs: llm)

    return use_llm


def test_conspect_mode_uses_conspect_prompt_and_suffix(audio, stub_pipeline):
    llm = FakeLLM("КОНСПЕКТ")
    stub_pipeline(llm)

    job = pipeline.run_file_job(audio, language="ru", mode=FormatMode.conspect)

    assert job.status.value == "completed"
    assert job.result is not None
    assert job.result.full_text == "КОНСПЕКТ"
    assert "конспект" in llm.prompts[0].lower()
    assert job.output_path is not None
    assert job.output_path.endswith("lecture.conspect.md")


def test_transcript_mode_uses_cleanup_prompt_and_plain_suffix(audio, stub_pipeline):
    llm = FakeLLM("ОЧИЩЕНО")
    stub_pipeline(llm)

    job = pipeline.run_file_job(audio, language="ru", mode=FormatMode.transcript)

    assert job.output_path is not None
    assert job.output_path.endswith("lecture.md")
    assert "исправь" in llm.prompts[0].lower()


def test_llm_failure_keeps_raw_transcript_and_warns(audio, stub_pipeline):
    stub_pipeline(BrokenLLM())

    job = pipeline.run_file_job(audio, language="ru", mode=FormatMode.conspect)

    assert job.status.value == "completed"
    assert job.result is not None
    assert job.result.full_text.startswith("Первое предложение")
    assert job.warnings and "llm unreachable" in job.warnings[0]


def test_no_llm_skips_post_processing(audio, stub_pipeline):
    llm = FakeLLM()
    stub_pipeline(llm)

    job = pipeline.run_file_job(audio, use_llm=False)

    assert llm.prompts == []
    assert job.result is not None
    assert job.result.full_text.startswith("Первое предложение")


def test_stt_failure_records_typed_error(audio, stub_pipeline, monkeypatch):
    class ExplodingTranscriber(FakeTranscriber):
        def transcribe(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("bad audio")

    stub_pipeline(FakeLLM())
    monkeypatch.setattr(pipeline, "WhisperTranscriber", ExplodingTranscriber)

    job = pipeline.run_file_job(audio)

    assert job.status.value == "failed"
    assert job.error is not None
    assert job.error.startswith("ValueError:")
    assert job.finished_at is not None


def test_progress_is_monotonic_and_complete(audio, stub_pipeline):
    stub_pipeline(FakeLLM())
    seen: list[tuple[str, float]] = []

    pipeline.run_file_job(audio, language="ru", on_progress=lambda s, p: seen.append((s, p)))

    percents = [p for _, p in seen]
    assert percents == sorted(percents)
    assert seen[0] == ("init", 0.0)
    assert seen[-1] == ("done", 1.0)
