"""Stage 4b - Retriever.

Input : a question (text).
Output: the top-k most relevant chunks, each with its source and a score.

It is the only place that connects the embedder to the vector store.
"""

from __future__ import annotations

from .embedder import Embedder
from .errors import VectorStoreError
from .vector_store import SearchHit, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        # Questions MUST be embedded with the same model that embedded the chunks;
        # vectors from different models live in unrelated "spaces".
        if embedder.model_name != store.embedding_model:
            raise VectorStoreError(
                f"The index was built with '{store.embedding_model}' but the current "
                f"embedding model is '{embedder.model_name}'. Re-run:  python main.py ingest"
            )
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k)
