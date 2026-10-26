"""Ties the stages together.

    ingest():   load -> chunk -> embed -> build index -> save to disk
    retrieve(): question -> embed -> nearest chunks
    generate_answer(): question + chunks -> prompt -> LLM -> answer

Retrieval and generation are separate methods on purpose. The generator is
passed in from outside, so replacing Gemini with a gateway later means giving
the pipeline a different Generator object. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import Chunk, chunk_document
from .config import Settings
from .embedder import Embedder
from .errors import ConfigError, IngestError, VectorStoreError
from .llm import Generator
from .loader import compute_manifest, load_documents
from .prompt import build_prompt, cited_numbers
from .retriever import Retriever
from .vector_store import SearchHit, VectorStore


@dataclass
class IngestReport:
    documents: int
    chunks: int
    embedding_model: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[SearchHit]
    cited: list[SearchHit]       # the retrieved chunks the answer actually cites
    provider: str | None         # which generator produced it (None if no LLM was called)


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, generator: Generator | None = None):
        self.settings = settings
        self.embedder = embedder
        self.generator = generator
        self.store: VectorStore | None = None
        self.retriever: Retriever | None = None

    # ---- indexing -------------------------------------------------------

    def ingest(self) -> IngestReport:
        loaded = load_documents(self.settings.docs_dir)
        if not loaded.documents:
            raise IngestError(
                f"No usable documents in {self.settings.docs_dir}.\n"
                + "\n".join(loaded.warnings)
                + "\nAdd .txt or text-based .pdf files and run ingestion again."
            )

        chunks: list[Chunk] = []
        for doc in loaded.documents:
            chunks.extend(chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap))
        if not chunks:
            raise IngestError("The documents produced no text chunks.")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        store = VectorStore.build(
            chunks, vectors, self.embedder.model_name, compute_manifest(self.settings.docs_dir)
        )
        store.save(self.settings.index_dir)
        self._use_store(store)
        return IngestReport(
            documents=len(loaded.documents),
            chunks=len(chunks),
            embedding_model=self.embedder.model_name,
            warnings=loaded.warnings,
        )

    def load_index(self) -> None:
        """Load the saved index from disk. Raises IndexNotFoundError if there is none."""
        self._use_store(VectorStore.load(self.settings.index_dir))

    def is_index_stale(self) -> bool:
        """True if the documents folder changed since the index was built."""
        if self.store is None:
            return False
        return compute_manifest(self.settings.docs_dir) != self.store.manifest

    def _use_store(self, store: VectorStore) -> None:
        self.store = store
        self.retriever = Retriever(self.embedder, store)

    # ---- answering ------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        question = (question or "").strip()
        if not question:
            raise ValueError("The question is empty.")
        if self.retriever is None:
            raise VectorStoreError("The index is not loaded. Call load_index() or ingest() first.")
        return self.retriever.retrieve(question, top_k or self.settings.top_k)

    def generate_answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(
                question=question,
                text="No relevant passages were found in the indexed documents.",
                hits=[], cited=[], provider=None,
            )
        if self.generator is None:
            raise ConfigError("No LLM generator is configured for this pipeline.")

        prompt = build_prompt(question, hits)
        text = self.generator.generate(prompt)
        cited = [hits[n - 1] for n in cited_numbers(text, len(hits))]
        return Answer(question=question, text=text, hits=hits, cited=cited, provider=self.generator.name)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Convenience: retrieve, then generate."""
        return self.generate_answer(question, self.retrieve(question, top_k))
