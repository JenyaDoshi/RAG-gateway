"""Stage 2 - Text chunker.

Input : one Document.
Output: a list of Chunk objects: overlapping pieces of the original text,
        each tagged with its source file and a chunk id.

Why chunk at all? See the README. Short version: embeddings work best on
focused passages, and sending only the relevant passages to the LLM is
cheaper, faster, and more accurate than sending whole documents.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "handbook.txt::3"
    source: str    # file the text came from
    index: int     # position of this chunk inside its file
    text: str


def _find_break(text: str, start: int, end: int, chunk_size: int) -> int:
    """Pick a natural place to cut, searching only the back half of the window.

    Preference order: paragraph break, sentence end, line break, space.
    """
    lo = start + chunk_size // 2

    paragraph = text.rfind("\n\n", lo, end)
    if paragraph != -1:
        return paragraph

    sentence = -1
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        pos = text.rfind(sep, lo, end)
        if pos != -1:
            sentence = max(sentence, pos + 1)  # keep the punctuation mark
    if sentence != -1:
        return sentence

    for sep in ("\n", " "):
        pos = text.rfind(sep, lo, end)
        if pos != -1:
            return pos

    return end  # no good boundary (e.g. one giant word): cut hard


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    min_chunk_chars: int = 100,
) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    n = len(text)
    chunks: list[str] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_break(text, start, end, chunk_size)
            if n - end < min_chunk_chars:
                end = n  # absorb a tiny tail instead of emitting a tiny chunk

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break

        # Next chunk starts `overlap` characters back, moved forward to a word start.
        next_start = end - overlap
        while 0 < next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = max(next_start, start + 1)  # always make progress

    return chunks


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc.source}::{i}", source=doc.source, index=i, text=piece)
        for i, piece in enumerate(chunk_text(doc.text, chunk_size, overlap))
    ]
