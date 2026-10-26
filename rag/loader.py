"""Stage 1 - Document loader.

Input : a folder of .txt / .md / .pdf files.
Output: a list of Document(source, text) with clean, normalized text,
        plus human-readable warnings for anything that was skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DocumentError

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str  # path relative to the docs folder, e.g. "handbook.pdf"
    text: str


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Make extracted text consistent without changing its wording."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)   # runs of spaces/tabs -> one space
    text = re.sub(r" ?\n ?", "\n", text)       # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)     # at most one blank line in a row
    return text.strip()


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")  # never fails, so it is the safe last resort


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported here so .txt-only users never load it

    reader = PdfReader(str(path))
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise DocumentError("the PDF is password-protected")
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _candidate_files(docs_dir: Path) -> list[Path]:
    return sorted(
        p for p in docs_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )


def _check_dir(docs_dir: Path) -> None:
    if not docs_dir.is_dir():
        raise DocumentError(
            f"Documents folder not found: {docs_dir}\n"
            "Create it and put your .txt / .pdf files inside."
        )


def compute_manifest(docs_dir: Path) -> dict[str, dict[str, int]]:
    """Fingerprint of the supported files (size + modified time).

    Saved next to the index so we can warn when documents changed after ingestion.
    """
    _check_dir(docs_dir)
    manifest = {}
    for path in _candidate_files(docs_dir):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            stat = path.stat()
            manifest[path.relative_to(docs_dir).as_posix()] = {
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
    return manifest


def load_documents(docs_dir: Path) -> LoadResult:
    _check_dir(docs_dir)
    result = LoadResult()

    for path in _candidate_files(docs_dir):
        name = path.relative_to(docs_dir).as_posix()
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            result.warnings.append(f"Skipped {name}: unsupported file type '{suffix or 'none'}'.")
            continue

        try:
            raw = _read_pdf(path) if suffix == ".pdf" else _read_text_file(path)
        except Exception as exc:  # any extraction failure should skip the file, not crash
            result.warnings.append(f"Skipped {name}: could not read it ({exc}).")
            continue

        text = normalize_text(raw)
        if not text:
            hint = " (a scanned PDF needs OCR, which this version does not do)" if suffix == ".pdf" else ""
            result.warnings.append(f"Skipped {name}: no text found{hint}.")
            continue

        result.documents.append(Document(source=name, text=text))

    return result
