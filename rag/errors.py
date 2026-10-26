"""All errors this project raises on purpose.

Every one of them inherits from RAGError, so the command-line entry point can
catch them in one place and print a friendly message instead of a traceback.
"""


class RAGError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(RAGError):
    """Settings are missing or invalid (for example, no API key)."""


class DocumentError(RAGError):
    """A document (or the documents folder) could not be read."""


class IngestError(RAGError):
    """Ingestion produced nothing usable (no documents, no text, no chunks)."""


class EmbeddingError(RAGError):
    """The embedding model could not be loaded or run."""


class VectorStoreError(RAGError):
    """The vector index is unusable (corrupt, or built with another model)."""


class IndexNotFoundError(VectorStoreError):
    """No saved index exists yet. Run ingestion first."""


class GenerationError(RAGError):
    """The LLM call failed.

    status_code and provider are kept on purpose: the LLM Gateway milestone
    will use them to decide between "fall back to another provider" and
    "stop, this error will not fix itself".
    """

    def __init__(self, message, status_code=None, provider=None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
