"""Offline stand-ins used by the tests (no model download, no API key, no network)."""

import re
import zlib

import numpy as np

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "at", "for", "with",
    "what", "which", "who", "how", "do", "does", "it", "that", "this", "be", "by", "or",
}


class FakeEmbedder:
    """Hashed bag-of-words vectors. Texts that share words get similar vectors.

    It is not smart like a real embedding model, but it is deterministic and
    good enough to test that the pipeline wiring works.
    """

    def __init__(self, dim=512, model_name="fake-hash-embedder"):
        self.dim = dim
        self.model_name = model_name
        self.doc_calls = 0
        self.query_calls = 0

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token not in _STOPWORDS:
                v[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        self.doc_calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)


class FakeGenerator:
    """Records the prompts it receives and returns a canned reply."""

    name = "fake"

    def __init__(self, reply="The answer is in the context [1]."):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply
