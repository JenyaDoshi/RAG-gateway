"""Stage 4a - Vector index.

Two different things are stored, side by side, in the index folder:

  index.faiss  - the VECTOR INDEX: only numbers (one embedding per chunk), laid
                 out so "which vectors are closest to this one?" is fast.
  chunks.json  - the ORIGINAL TEXT of every chunk plus its metadata. The index
                 answers "row 7 is closest"; this file says what row 7 says.

The two are linked by position: row i of the index is chunk i in chunks.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk
from .errors import IndexNotFoundError, VectorStoreError

INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity: higher means more similar


class VectorStore:
    def __init__(self, index, chunks: list[Chunk], embedding_model: str, manifest: dict):
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.manifest = manifest  # fingerprint of the documents this index was built from

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, embedding_model: str, manifest: dict):
        if len(chunks) == 0 or len(chunks) != len(vectors):
            raise VectorStoreError("Need one embedding per chunk (and at least one chunk).")
        # IndexFlatIP = exact search by inner product. With unit-length vectors
        # that is cosine similarity. Exact search is plenty fast for thousands of chunks.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(index, list(chunks), embedding_model, manifest)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if top_k < 1 or not self.chunks:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        if query.shape[1] != self.index.d:
            raise VectorStoreError(
                f"Question vector has {query.shape[1]} dimensions but the index has "
                f"{self.index.d}. The index was built with a different embedding model; "
                "re-run ingestion."
            )
        scores, ids = self.index.search(query, min(top_k, len(self.chunks)))
        return [
            SearchHit(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        meta = {
            "format": FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dimension": int(self.index.d),
            "manifest": self.manifest,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (directory / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        index_path, meta_path = directory / INDEX_FILE, directory / META_FILE
        if not index_path.is_file() or not meta_path.is_file():
            raise IndexNotFoundError(
                f"No index found in {directory}. Build it with:  python main.py ingest"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in meta["chunks"]]
            index = faiss.read_index(str(index_path))
            if index.ntotal != len(chunks):
                raise ValueError("index and chunk file disagree")
            return cls(index, chunks, meta["embedding_model"], meta.get("manifest", {}))
        except Exception as exc:
            raise VectorStoreError(
                f"The saved index looks corrupt ({exc}). Rebuild it with:  python main.py ingest"
            ) from exc
