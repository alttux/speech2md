from __future__ import annotations

import logging
import re
from typing import Callable

from speech2md.core.models import FormatMode
from speech2md.core.postprocess.llm_client import LLMClient
from speech2md.core.postprocess.prompts import get_prompt

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2000
OVERLAP = 200

MERGE_SEPARATOR = "\n\n---\n\n"

# Below this many characters a "match" between chunk boundaries is more likely
# coincidence (shared punctuation, a common word) than real duplicated text.
MIN_OVERLAP_MATCH = 24

ProgressCallback = Callable[[float], None]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n{2,}")


def _split_sentences(text: str, max_words: int) -> list[str]:
    """Split into sentences, hard-wrapping any sentence longer than max_words.

    Whisper output is often sparsely punctuated, so a single "sentence" can be
    arbitrarily long; without the wrap it would blow past the chunk budget.
    """
    sentences: list[str] = []
    for raw in _SENTENCE_BOUNDARY.split(text):
        piece = raw.strip()
        if not piece:
            continue
        words = piece.split()
        if len(words) <= max_words:
            sentences.append(piece)
        else:
            for i in range(0, len(words), max_words):
                sentences.append(" ".join(words[i : i + max_words]))
    return sentences


def build_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into chunks of at most chunk_size words, aligned to sentences."""
    stripped = text.strip()
    if not stripped:
        return []

    chunk_size = max(1, chunk_size)
    if len(stripped.split()) <= chunk_size:
        return [stripped]

    overlap = max(0, min(overlap, chunk_size // 2))
    sentences = _split_sentences(stripped, chunk_size)
    counts = [len(s.split()) for s in sentences]

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        total = 0
        end = start
        while end < len(sentences) and (end == start or total + counts[end] <= chunk_size):
            total += counts[end]
            end += 1

        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break

        back = end
        carried = 0
        while back > start + 1 and carried + counts[back - 1] <= overlap:
            back -= 1
            carried += counts[back]
        start = back

    return chunks


def format_text(text: str, llm: LLMClient, language: str = "", prompt_template: str = "") -> str:
    if not text.strip():
        return text

    prompt = get_prompt(language, prompt_template).format(text=text)
    return llm.complete(prompt)


def format_long_text(
    text: str,
    llm: LLMClient,
    language: str = "",
    prompt_template: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Clean up a transcript chunk by chunk, trimming duplicated overlap text."""
    chunks = build_chunks(text, chunk_size, overlap)
    if not chunks:
        return text
    if len(chunks) == 1:
        result = format_text(chunks[0], llm, language, prompt_template)
        if on_progress:
            on_progress(1.0)
        return result

    formatted: list[str] = []
    for i, chunk in enumerate(chunks):
        result = format_text(chunk, llm, language, prompt_template)
        if formatted:
            prev = formatted[-1]
            common = _find_overlap(prev, result, window=_overlap_window(overlap))
            formatted[-1] = prev[: -len(common)].rstrip() if common else prev
        formatted.append(result)
        if on_progress:
            on_progress((i + 1) / len(chunks))

    return "\n\n".join(part for part in formatted if part.strip())


def summarize_long_text(
    text: str,
    llm: LLMClient,
    language: str = "",
    prompt_template: str = "",
    merge_template: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Map-reduce summarization: summarize each chunk, then merge the summaries."""
    chunks = build_chunks(text, chunk_size, overlap)
    if not chunks:
        return text

    summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt = get_prompt(language, prompt_template, mode="conspect").format(text=chunk)
        summaries.append(llm.complete(prompt))
        if on_progress:
            on_progress(0.8 * (i + 1) / len(chunks))

    merged = _merge_summaries(summaries, llm, language, merge_template, chunk_size)
    if on_progress:
        on_progress(1.0)
    return merged


def _merge_summaries(
    summaries: list[str],
    llm: LLMClient,
    language: str,
    merge_template: str,
    chunk_size: int,
) -> str:
    parts = [s.strip() for s in summaries if s.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    while len(parts) > 1:
        groups = _group_by_budget(parts, chunk_size)
        if len(groups) == len(parts):
            # Every summary already fills the budget on its own; another merge
            # pass would only truncate, so keep them side by side.
            logger.warning(
                "Summaries too large to merge into one pass — joining %d parts as-is",
                len(parts),
            )
            return MERGE_SEPARATOR.join(parts)

        merged: list[str] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            prompt = get_prompt(language, merge_template, mode="merge").format(
                text=MERGE_SEPARATOR.join(group)
            )
            merged.append(llm.complete(prompt).strip())
        parts = merged

    return parts[0]


def _group_by_budget(parts: list[str], chunk_size: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for part in parts:
        words = len(part.split())
        if current and current_words + words > chunk_size:
            groups.append(current)
            current = []
            current_words = 0
        current.append(part)
        current_words += words

    if current:
        groups.append(current)
    return groups


def _overlap_window(overlap: int) -> int:
    # overlap is a word count; _find_overlap works on characters.
    return max(OVERLAP, overlap * 8)


def process_text(
    text: str,
    llm: LLMClient,
    *,
    mode: FormatMode = FormatMode.transcript,
    language: str = "",
    prompt_template: str = "",
    merge_template: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    on_progress: ProgressCallback | None = None,
) -> str:
    if mode == FormatMode.conspect:
        return summarize_long_text(
            text,
            llm,
            language=language,
            prompt_template=prompt_template,
            merge_template=merge_template,
            chunk_size=chunk_size,
            overlap=overlap,
            on_progress=on_progress,
        )
    return format_long_text(
        text,
        llm,
        language=language,
        prompt_template=prompt_template,
        chunk_size=chunk_size,
        overlap=overlap,
        on_progress=on_progress,
    )


def _find_overlap(a: str, b: str, window: int = 200, min_match: int = MIN_OVERLAP_MATCH) -> str:
    a_end = a[-window:].strip()
    b_start = b[:window].strip()
    for i in range(min(len(a_end), len(b_start)), min_match - 1, -1):
        if a_end[-i:].strip() == b_start[:i].strip():
            return a_end[-i:]
    return ""
