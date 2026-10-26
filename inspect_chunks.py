"""Print how your documents are split into chunks (no embeddings, no API key).

    python inspect_chunks.py

Use it to check that chunks look like sensible, self-contained passages.
"""

from rag.chunker import chunk_document
from rag.config import load_settings
from rag.errors import RAGError
from rag.loader import load_documents


def main() -> int:
    try:
        settings = load_settings()
        loaded = load_documents(settings.docs_dir)
    except RAGError as exc:
        print(f"Error: {exc}")
        return 1

    for warning in loaded.warnings:
        print(f"warning: {warning}")
    total = 0
    for doc in loaded.documents:
        chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        total += len(chunks)
        print(f"\n=== {doc.source}: {len(doc.text)} characters -> {len(chunks)} chunks ===")
        for chunk in chunks:
            preview = chunk.text.replace("\n", " ")
            print(f"[{chunk.chunk_id}] ({len(chunk.text)} chars) {preview[:100]}...")
    print(f"\nTotal: {len(loaded.documents)} document(s), {total} chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
