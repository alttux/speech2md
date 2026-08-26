from speech2md.core.models import Job, JobMode, Segment, Settings, TranscriptionResult


def test_segment() -> None:
    s = Segment(start=0.0, end=2.5, text="hello world", language="en", confidence=0.95)
    assert s.text == "hello world"
    assert s.confidence == 0.95


def test_transcription_result() -> None:
    segs = [Segment(start=0.0, end=1.0, text="one"), Segment(start=1.0, end=2.0, text="two")]
    r = TranscriptionResult(segments=segs, full_text="one two", language="en", duration=2.0)
    assert len(r.segments) == 2
    assert r.full_text == "one two"


def test_job_defaults() -> None:
    j = Job(source="test.mp3", mode=JobMode.file)
    assert j.status.value == "pending"
    assert len(j.id) == 12


def test_settings_defaults() -> None:
    s = Settings()
    assert s.whisper_model == "small"
    assert s.device.value == "cpu"
    assert s.llm_backend.value == "ollama"


def test_config_roundtrip(tmp_path, monkeypatch) -> None:
    from speech2md.core.config import load, save

    monkeypatch.setattr("speech2md.core.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("speech2md.core.config.CONFIG_FILE", tmp_path / "config.toml")

    s = Settings(whisper_model="medium", device="cpu", language="ru")
    save(s)

    loaded = load()
    assert loaded.whisper_model == "medium"
    assert loaded.language == "ru"
