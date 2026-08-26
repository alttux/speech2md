import pytest

from speech2md.core.models import FormatMode
from speech2md.core.postprocess.formatter import (
    build_chunks,
    format_long_text,
    process_text,
    summarize_long_text,
)
from speech2md.core.postprocess.llm_client import LLMClient


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return prompt


def test_short_text_is_one_chunk() -> None:
    assert build_chunks("One two three.", chunk_size=100) == ["One two three."]


def test_empty_text_yields_no_chunks() -> None:
    assert build_chunks("   \n  ", chunk_size=10) == []


def test_chunks_split_on_sentence_boundaries() -> None:
    text = "One two three four five. Six seven eight nine ten."
    assert build_chunks(text, chunk_size=5, overlap=0) == [
        "One two three four five.",
        "Six seven eight nine ten.",
    ]


def test_chunks_respect_word_budget() -> None:
    text = " ".join(f"word{i}." for i in range(60))
    for chunk in build_chunks(text, chunk_size=10, overlap=2):
        assert len(chunk.split()) <= 10


def test_unpunctuated_text_is_hard_wrapped() -> None:
    # Whisper often returns long runs with no sentence marks at all.
    text = " ".join(f"w{i}" for i in range(100))
    chunks = build_chunks(text, chunk_size=10, overlap=0)
    assert len(chunks) == 10
    assert all(len(c.split()) == 10 for c in chunks)


def test_overlap_larger_than_chunk_size_still_terminates() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    chunks = build_chunks(text, chunk_size=10, overlap=999)
    assert 0 < len(chunks) < 200


def test_chunk_size_is_measured_in_words_not_characters() -> None:
    # 50 words is well under the 100-word budget but far over 100 characters.
    text = " ".join(f"word{i}" for i in range(50))
    assert len(text) > 100
    assert build_chunks(text, chunk_size=100) == [text]


def test_format_long_text_single_chunk_makes_one_call() -> None:
    llm = FakeLLM(["cleaned"])
    assert format_long_text("short text", llm, chunk_size=100) == "cleaned"
    assert len(llm.prompts) == 1


def test_format_long_text_trims_duplicated_overlap() -> None:
    llm = FakeLLM(
        [
            "Alpha beta gamma. The quick brown fox jumps over.",
            "The quick brown fox jumps over. Delta epsilon.",
        ]
    )
    text = "One two three four five. Six seven eight nine ten."
    result = format_long_text(text, llm, chunk_size=5, overlap=0)

    assert result == "Alpha beta gamma.\n\nThe quick brown fox jumps over. Delta epsilon."


def test_format_long_text_reports_progress() -> None:
    seen: list[float] = []
    llm = FakeLLM(["a", "b"])
    text = "One two three four five. Six seven eight nine ten."
    format_long_text(text, llm, chunk_size=5, overlap=0, on_progress=seen.append)

    assert seen == [0.5, 1.0]


def test_summarize_maps_each_chunk_then_merges() -> None:
    llm = FakeLLM(["A", "B", "C", "MERGED"])
    text = (
        "One two three four five. "
        "Six seven eight nine ten. "
        "Eleven twelve thirteen fourteen fifteen."
    )
    result = summarize_long_text(text, llm, chunk_size=5, overlap=0)

    assert result == "MERGED"
    assert len(llm.prompts) == 4
    assert "A" in llm.prompts[-1] and "B" in llm.prompts[-1] and "C" in llm.prompts[-1]


def test_summarize_single_chunk_skips_merge() -> None:
    llm = FakeLLM(["SUMMARY"])
    assert summarize_long_text("short text", llm, chunk_size=100) == "SUMMARY"
    assert len(llm.prompts) == 1


def test_summarize_uses_conspect_prompt() -> None:
    llm = FakeLLM(["SUMMARY"])
    summarize_long_text("short text", llm, language="ru", chunk_size=100)
    assert "конспект" in llm.prompts[0].lower()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(FormatMode.transcript, "CLEANED"), (FormatMode.conspect, "SUMMARY")],
)
def test_process_text_dispatches_on_mode(mode: FormatMode, expected: str) -> None:
    llm = FakeLLM([expected])
    assert process_text("short text", llm, mode=mode, chunk_size=100) == expected
