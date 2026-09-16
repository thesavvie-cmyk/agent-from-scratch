"""Text chunking utilities (block 10).

Two strategies
--------------
fixed_length_chunking   — splits by character count (fast, simple)
sentence_aware_chunking — respects sentence boundaries (better recall)
"""
from __future__ import annotations

import re


def fixed_length_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into fixed-length character chunks with overlap.

    Parameters
    ----------
    text:
        Input text.
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Characters shared between consecutive chunks.
        Must be strictly less than *chunk_size* — equal or greater would
        cause the sliding window to stop advancing (infinite loop).

    Returns
    -------
    list[str]
        Non-empty chunks.  Empty input returns [].

    Raises
    ------
    ValueError
        If overlap >= chunk_size.
    """
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def sentence_aware_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into chunks that prefer sentence boundaries.

    Sentences are detected by ``. ! ?`` followed by whitespace or end-of-string.
    Chunks accumulate whole sentences until the next sentence would overflow
    *chunk_size*.  Overlap is implemented by re-including trailing sentences
    that collectively fit within *overlap* characters.

    For sentences longer than *chunk_size*, falls back to character splitting
    so output is always bounded.

    Parameters
    ----------
    text, chunk_size, overlap:
        Same semantics as fixed_length_chunking.

    Raises
    ------
    ValueError
        If overlap >= chunk_size.
    """
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    def _flush_and_overlap() -> tuple[list[str], int]:
        """Flush current chunk; return overlap sentences for next chunk."""
        chunks.append(" ".join(current))
        overlap_sents: list[str] = []
        overlap_len = 0
        for s in reversed(current):
            sep = 1 if overlap_sents else 0
            if overlap_len + sep + len(s) > overlap:
                break
            overlap_sents.insert(0, s)
            overlap_len += sep + len(s)
        return overlap_sents, overlap_len

    for sent in sentences:
        sep = 1 if current else 0
        if current_len + sep + len(sent) <= chunk_size:
            current.append(sent)
            current_len += sep + len(sent)
        else:
            if current:
                current, current_len = _flush_and_overlap()

            sep = 1 if current else 0
            if current_len + sep + len(sent) <= chunk_size:
                current.append(sent)
                current_len += sep + len(sent)
            else:
                # Sentence longer than chunk_size — character-split it
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                step = chunk_size - overlap
                for i in range(0, len(sent), step):
                    sub = sent[i : i + chunk_size]
                    if sub:
                        chunks.append(sub)

    if current:
        chunks.append(" ".join(current))

    return chunks
