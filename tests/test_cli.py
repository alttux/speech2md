import pytest
from typer.testing import CliRunner

from speech2md.cli.app import app
from speech2md.core import pipeline
from speech2md.core.models import Segment, Settings, TranscriptionResult
from speech2md.core.postprocess.llm_client import LLMClient

runner = CliRunner()


class FakeTranscriber:
    def __init__(self, **kwargs: object) -> None:
        pass

    def transcribe(self, audio_path, language="", on_progress=None):  # type: ignore[no-untyped-def]
        if on_progress:
            on_progress(1.0)
        return TranscriptionResult(
            segments=[Segment(start=0.0, end=1.0, text="привет")],
            full_text="Привет мир. Как дела сегодня.",
            language="ru",
            duration=1.0,
        )


class FakeLLM(LLMClient):
    def __init__(self, response: str = "PROCESSED") -> None:
        self.response = response

    def complete(self, prompt: str) -> str:
        return self.response


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
    monkeypatch.setattr(pipeline, "create_client", lambda **kwargs: FakeLLM())
    return settings


def test_transcribe_also_conspect_produces_two_files(audio, stub_pipeline, tmp_path):
    result = runner.invoke(app, ["transcribe", str(audio), "--also-conspect"])

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "out"
    assert (out_dir / "lecture.md").exists()
    assert (out_dir / "lecture.conspect.md").exists()
    assert "Transcription saved" in result.output
    assert "Conspect saved" in result.output


def test_also_conspect_rejects_no_llm(audio, stub_pipeline):
    result = runner.invoke(app, ["transcribe", str(audio), "--also-conspect", "--no-llm"])

    assert result.exit_code == 1
    assert "--also-conspect requires LLM" in result.output


def test_also_conspect_rejects_explicit_conspect_mode(audio, stub_pipeline):
    result = runner.invoke(
        app, ["transcribe", str(audio), "--also-conspect", "--mode", "conspect"]
    )

    assert result.exit_code == 1
    assert "already builds a conspect" in result.output


def test_transcribe_without_also_conspect_writes_single_file(audio, stub_pipeline, tmp_path):
    result = runner.invoke(app, ["transcribe", str(audio)])

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "out"
    assert (out_dir / "lecture.md").exists()
    assert not (out_dir / "lecture.conspect.md").exists()
