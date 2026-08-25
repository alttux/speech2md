from speech2md.core.export import write_srt
from speech2md.core.models import Segment, TranscriptionResult


def test_write_srt_format(tmp_path) -> None:
    result = TranscriptionResult(
        segments=[
            Segment(start=0.0, end=1.5, text="hello world"),
            Segment(start=61.25, end=63.0, text="second line"),
        ],
        full_text="hello world second line",
        language="en",
    )

    path = write_srt(result, tmp_path / "out.srt")
    content = path.read_text(encoding="utf-8")

    assert content == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "hello world\n"
        "\n"
        "2\n"
        "00:01:01,250 --> 00:01:03,000\n"
        "second line\n"
    )


def test_write_srt_creates_parent_dir(tmp_path) -> None:
    result = TranscriptionResult(segments=[], full_text="")
    path = write_srt(result, tmp_path / "nested" / "out.srt")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
