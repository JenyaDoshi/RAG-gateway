"""Offline tests: run with   python -m unittest discover -s tests -v

They check the wiring and logic of every stage using fake embeddings and a fake
LLM. They do NOT check retrieval quality of the real embedding model or the
real Gemini API; use `python main.py search ...` / `ask ...` for that.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as cli
from rag.chunker import chunk_document, chunk_text
from rag.config import Settings
from rag.errors import (ConfigError, DocumentError, GenerationError, IndexNotFoundError,
                        IngestError, VectorStoreError)
from rag.llm import GeminiGenerator
from rag.loader import Document, load_documents, normalize_text
from rag.pipeline import RAGPipeline
from rag.prompt import Prompt, build_prompt, cited_numbers
from rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder, FakeGenerator

SAMPLE_DOCS = {
    "drone.txt": ("The supply drone lands on the roof pad every Thursday at 14:20. "
                  "If wind exceeds 40 km/h the landing is postponed and logistics is notified. " * 6),
    "cooking.txt": ("Sourdough bread needs flour, water, salt and a live starter. "
                    "Let the dough rise overnight before baking in a hot oven. " * 6),
    "space.txt": ("Jupiter is the largest planet in the solar system and has dozens of moons. "
                  "Its Great Red Spot is a storm larger than Earth. " * 6),
}


def make_settings(root: Path, **overrides) -> Settings:
    values = dict(
        docs_dir=root / "docs", index_dir=root / "index", model_cache_dir=root / "models",
        embedding_model="fake-hash-embedder", gemini_model="test-model", gemini_api_key="test-key",
        chunk_size=300, chunk_overlap=60, top_k=3, llm_timeout_s=5,
    )
    values.update(overrides)
    return Settings(**values)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "docs").mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write_docs(self, docs=None):
        for name, text in (docs or SAMPLE_DOCS).items():
            (self.root / "docs" / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- loader
class LoaderTests(TempDirCase):
    def test_loads_txt_and_md_and_normalizes(self):
        (self.root / "docs" / "a.txt").write_text("Hello   world\r\n\r\n\r\n\r\nSecond  line", encoding="utf-8")
        (self.root / "docs" / "b.md").write_text("# Title\nBody", encoding="utf-8")
        result = load_documents(self.root / "docs")
        self.assertEqual([d.source for d in result.documents], ["a.txt", "b.md"])
        self.assertEqual(result.documents[0].text, "Hello world\n\nSecond line")
        self.assertEqual(result.warnings, [])

    def test_empty_and_unsupported_files_are_skipped_with_warnings(self):
        (self.root / "docs" / "empty.txt").write_text("   \n  ", encoding="utf-8")
        (self.root / "docs" / "image.png").write_bytes(b"\x89PNG")
        (self.root / "docs" / "ok.txt").write_text("content", encoding="utf-8")
        result = load_documents(self.root / "docs")
        self.assertEqual([d.source for d in result.documents], ["ok.txt"])
        self.assertEqual(len(result.warnings), 2)

    def test_missing_folder_raises_clear_error(self):
        with self.assertRaises(DocumentError):
            load_documents(self.root / "does-not-exist")

    def test_non_utf8_text_file_still_loads(self):
        (self.root / "docs" / "latin.txt").write_bytes("caf\xe9 au lait".encode("latin-1"))
        result = load_documents(self.root / "docs")
        self.assertIn("caf", result.documents[0].text)

    def test_corrupt_pdf_is_skipped_not_fatal(self):
        (self.root / "docs" / "broken.pdf").write_bytes(b"this is not a pdf")
        result = load_documents(self.root / "docs")
        self.assertEqual(result.documents, [])
        self.assertEqual(len(result.warnings), 1)

    def test_real_pdf_text_is_extracted(self):
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed (only needed to build a test PDF)")
        path = self.root / "docs" / "report.pdf"
        c = canvas.Canvas(str(path))
        c.drawString(72, 750, "The quarterly report mentions Project Falcon.")
        c.save()
        result = load_documents(self.root / "docs")
        self.assertEqual(len(result.documents), 1)
        self.assertIn("Project Falcon", result.documents[0].text)

    def test_normalize_keeps_words_intact(self):
        self.assertEqual(normalize_text("  a \t b\x00 "), "a b")


# --------------------------------------------------------------- chunker
class ChunkerTests(unittest.TestCase):
    TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(120))

    def test_empty_inputs_give_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("Just a short note."), ["Just a short note."])

    def test_chunks_are_exact_substrings_of_the_original(self):
        for chunk in chunk_text(self.TEXT, 200, 40):
            self.assertIn(chunk, self.TEXT)

    def test_size_limit_and_no_tiny_or_empty_chunks(self):
        chunks = chunk_text(self.TEXT, 200, 40, min_chunk_chars=60)
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(chunk.strip())
            self.assertLessEqual(len(chunk), 200 + 60)
            self.assertGreaterEqual(len(chunk), 60)

    def test_neighbouring_chunks_overlap(self):
        chunks = chunk_text(self.TEXT, 200, 60)
        for a, b in zip(chunks, chunks[1:]):
            self.assertIn(b[:15], a)

    def test_every_word_is_covered(self):
        chunks = chunk_text(self.TEXT, 200, 40)
        joined = " ".join(chunks)
        for i in range(120):
            self.assertIn(f"Sentence number {i} ", joined)

    def test_text_without_spaces_still_terminates(self):
        chunks = chunk_text("x" * 2000, 300, 50)
        self.assertGreater(len(chunks), 1)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 100, 100)
        with self.assertRaises(ValueError):
            chunk_text("abc", 0, 0)

    def test_chunk_metadata(self):
        chunks = chunk_document(Document("notes.txt", self.TEXT), 300, 50)
        self.assertEqual(chunks[0].chunk_id, "notes.txt::0")
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(c.source == "notes.txt" for c in chunks))


# ---------------------------------------------------------- prompt/citations
class PromptTests(TempDirCase):
    def test_prompt_contains_question_numbered_context_and_sources(self):
        self.write_docs()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder())
        pipe.ingest()
        hits = pipe.retrieve("When does the drone land?")
        prompt = build_prompt("When does the drone land?", hits)
        self.assertIsInstance(prompt, Prompt)
        self.assertIn("Question: When does the drone land?", prompt.user)
        self.assertIn("[1] (source:", prompt.user)
        self.assertIn(hits[0].chunk.text, prompt.user)
        self.assertIn("ONLY", prompt.system)

    def test_cited_numbers_parsing(self):
        self.assertEqual(cited_numbers("A [2] and B [1, 3] and again [2].", 3), [2, 1, 3])
        self.assertEqual(cited_numbers("Nothing cited here.", 3), [])
        self.assertEqual(cited_numbers("Out of range [9] and [0].", 3), [])


# ------------------------------------------------------------ vector store
class VectorStoreTests(TempDirCase):
    def test_save_load_round_trip_and_ranking(self):
        self.write_docs()
        embedder = FakeEmbedder()
        pipe = RAGPipeline(make_settings(self.root), embedder)
        report = pipe.ingest()
        self.assertEqual(report.documents, 3)
        self.assertGreaterEqual(report.chunks, 3)

        loaded = VectorStore.load(self.root / "index")
        self.assertEqual(len(loaded.chunks), report.chunks)
        hits = loaded.search(embedder.embed_query("sourdough starter flour"), 3)
        self.assertEqual(hits[0].chunk.source, "cooking.txt")
        self.assertGreaterEqual(hits[0].score, hits[-1].score)

    def test_missing_index_raises_index_not_found(self):
        with self.assertRaises(IndexNotFoundError):
            VectorStore.load(self.root / "nothing-here")

    def test_corrupt_index_raises_store_error(self):
        self.write_docs()
        RAGPipeline(make_settings(self.root), FakeEmbedder()).ingest()
        (self.root / "index" / "chunks.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(VectorStoreError):
            VectorStore.load(self.root / "index")

    def test_top_k_larger_than_index_is_fine(self):
        self.write_docs({"one.txt": "just one small chunk of text"})
        embedder = FakeEmbedder()
        pipe = RAGPipeline(make_settings(self.root), embedder)
        pipe.ingest()
        self.assertEqual(len(pipe.retrieve("small chunk", top_k=10)), 1)


# ---------------------------------------------------------------- pipeline
class PipelineTests(TempDirCase):
    def test_retrieval_finds_the_right_document(self):
        self.write_docs()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder())
        pipe.ingest()
        self.assertEqual(pipe.retrieve("when does the supply drone land")[0].chunk.source, "drone.txt")
        self.assertEqual(pipe.retrieve("largest planet jupiter moons")[0].chunk.source, "space.txt")

    def test_index_is_reused_without_re_embedding_documents(self):
        self.write_docs()
        RAGPipeline(make_settings(self.root), FakeEmbedder()).ingest()

        second_embedder = FakeEmbedder()
        second = RAGPipeline(make_settings(self.root), second_embedder)
        second.load_index()
        second.retrieve("drone")
        second.retrieve("bread")
        self.assertEqual(second_embedder.doc_calls, 0)   # no document was re-embedded
        self.assertEqual(second_embedder.query_calls, 2)

    def test_multiple_questions_reuse_the_same_index(self):
        self.write_docs()
        embedder, gen = FakeEmbedder(), FakeGenerator("Answer [1].")
        pipe = RAGPipeline(make_settings(self.root), embedder, gen)
        pipe.ingest()
        docs_calls = embedder.doc_calls
        for q in ("drone landing", "sourdough bread", "jupiter moons"):
            self.assertTrue(pipe.ask(q).text)
        self.assertEqual(embedder.doc_calls, docs_calls)
        self.assertEqual(len(gen.prompts), 3)

    def test_ask_returns_answer_with_cited_sources(self):
        self.write_docs()
        gen = FakeGenerator("The drone lands at 14:20 [1].")
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder(), gen)
        pipe.ingest()
        answer = pipe.ask("when does the supply drone land")
        self.assertEqual(answer.provider, "fake")
        self.assertEqual([h.chunk.source for h in answer.cited], ["drone.txt"])
        self.assertIn("when does the supply drone land", gen.prompts[0].user)

    def test_refusal_has_no_citations(self):
        self.write_docs()
        gen = FakeGenerator("I don't have enough information in the provided documents to answer that.")
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder(), gen)
        pipe.ingest()
        self.assertEqual(pipe.ask("what is the wifi password").cited, [])

    def test_no_hits_skips_the_llm(self):
        gen = FakeGenerator()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder(), gen)
        answer = pipe.generate_answer("anything", [])
        self.assertIn("No relevant passages", answer.text)
        self.assertEqual(gen.prompts, [])

    def test_empty_question_is_rejected(self):
        self.write_docs()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder(), FakeGenerator())
        pipe.ingest()
        with self.assertRaises(ValueError):
            pipe.retrieve("   ")

    def test_empty_docs_folder_raises_ingest_error(self):
        with self.assertRaises(IngestError):
            RAGPipeline(make_settings(self.root), FakeEmbedder()).ingest()

    def test_only_empty_files_raises_ingest_error(self):
        self.write_docs({"blank.txt": "   "})
        with self.assertRaises(IngestError):
            RAGPipeline(make_settings(self.root), FakeEmbedder()).ingest()

    def test_index_built_with_another_model_is_rejected(self):
        self.write_docs()
        RAGPipeline(make_settings(self.root), FakeEmbedder(model_name="model-A")).ingest()
        other = RAGPipeline(make_settings(self.root), FakeEmbedder(model_name="model-B"))
        with self.assertRaises(VectorStoreError):
            other.load_index()

    def test_staleness_detected_when_documents_change(self):
        self.write_docs()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder())
        pipe.ingest()
        self.assertFalse(pipe.is_index_stale())
        (self.root / "docs" / "new.txt").write_text("a brand new document", encoding="utf-8")
        self.assertTrue(pipe.is_index_stale())

    def test_missing_generator_is_a_config_error(self):
        self.write_docs()
        pipe = RAGPipeline(make_settings(self.root), FakeEmbedder())
        pipe.ingest()
        with self.assertRaises(ConfigError):
            pipe.ask("drone")


# ------------------------------------------------------------------ gemini
class _FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


class _FakeClient:
    def __init__(self, behaviour):
        self.models = _FakeModels(behaviour)


class _Resp:
    def __init__(self, text):
        self.text = text
        self.candidates = []


class GeminiGeneratorTests(unittest.TestCase):
    PROMPT = Prompt(system="be grounded", user="Context... Question: hi")

    def gen(self, behaviour):
        client = _FakeClient(behaviour)
        return GeminiGenerator("key", "test-model", client=client), client

    def test_success_passes_prompt_and_model(self):
        gen, client = self.gen(_Resp("  hello [1]  "))
        self.assertEqual(gen.generate(self.PROMPT), "hello [1]")
        call = client.models.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["contents"], self.PROMPT.user)
        self.assertEqual(call["config"].system_instruction, "be grounded")

    def test_http_errors_become_generation_errors_with_status(self):
        from google.genai import errors
        cases = {
            429: errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
            403: errors.ClientError(403, {"error": {"message": "denied", "status": "PERMISSION_DENIED"}}),
            400: errors.ClientError(400, {"error": {"message": "API key not valid", "status": "INVALID_ARGUMENT"}}),
            404: errors.ClientError(404, {"error": {"message": "no model", "status": "NOT_FOUND"}}),
            503: errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}),
        }
        for code, exc in cases.items():
            gen, _ = self.gen(exc)
            with self.assertRaises(GenerationError) as ctx:
                gen.generate(self.PROMPT)
            self.assertEqual(ctx.exception.status_code, code)
            self.assertEqual(ctx.exception.provider, "gemini")

    def test_network_failure_becomes_generation_error(self):
        gen, _ = self.gen(ConnectionError("offline"))
        with self.assertRaises(GenerationError) as ctx:
            gen.generate(self.PROMPT)
        self.assertIsNone(ctx.exception.status_code)

    def test_empty_response_is_an_error(self):
        gen, _ = self.gen(_Resp(None))
        with self.assertRaises(GenerationError):
            gen.generate(self.PROMPT)


# --------------------------------------------------------------- config/cli
class ConfigTests(unittest.TestCase):
    def test_missing_or_placeholder_key_is_rejected(self):
        base = make_settings(Path("."), gemini_api_key=None)
        with self.assertRaises(ConfigError):
            base.require_api_key()
        with self.assertRaises(ConfigError):
            make_settings(Path("."), gemini_api_key="your-api-key-here").require_api_key()
        self.assertEqual(make_settings(Path("."), gemini_api_key=" abc ").require_api_key(), "abc")

    def test_bad_env_values_are_rejected(self):
        from rag.config import load_settings
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": "abc"}):
            with self.assertRaises(ConfigError):
                load_settings()
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}):
            with self.assertRaises(ConfigError):
                load_settings()


class CliTests(TempDirCase):
    """Runs the real command-line code paths with fake embedder/generator."""

    def run_cli(self, argv, *, generator=None, inputs=None, key="test-key"):
        env = {"DOCS_DIR": str(self.root / "docs"), "INDEX_DIR": str(self.root / "index"),
               "MODEL_CACHE_DIR": str(self.root / "models"), "EMBEDDING_MODEL": "fake-hash-embedder"}
        if key is not None:
            env["GEMINI_API_KEY"] = key
        out, err = io.StringIO(), io.StringIO()
        patches = [
            mock.patch.dict(os.environ, env),
            mock.patch.object(cli, "make_embedder", lambda s: FakeEmbedder()),
            mock.patch.object(cli, "make_generator",
                              (lambda s: (s.require_api_key(), generator or FakeGenerator())[1])),
        ]
        if inputs is not None:
            patches.append(mock.patch("builtins.input", side_effect=inputs))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            os.environ.pop("GEMINI_API_KEY", None) if key is None else None
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_ingest_search_ask_flow(self):
        self.write_docs()
        code, out, _ = self.run_cli(["ingest"])
        self.assertEqual(code, 0)
        self.assertIn("Indexed 3 document(s)", out)

        code, out, _ = self.run_cli(["search", "when does the supply drone land"])
        self.assertEqual(code, 0)
        self.assertIn("drone.txt", out)

        code, out, _ = self.run_cli(["ask", "when does the supply drone land"],
                                    generator=FakeGenerator("At 14:20 [1]."))
        self.assertEqual(code, 0)
        self.assertIn("At 14:20 [1].", out)
        self.assertIn("Sources cited:", out)
        self.assertLess(out.index("Retrieved passages"), out.index("Answer:"))  # chunks shown first

    def test_ask_auto_builds_index_on_first_run(self):
        self.write_docs()
        code, out, _ = self.run_cli(["ask", "jupiter moons"])
        self.assertEqual(code, 0)
        self.assertIn("building it", out)

    def test_chat_handles_several_questions_and_survives_an_llm_error(self):
        self.write_docs()

        class Flaky(FakeGenerator):
            def generate(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 2:
                    raise GenerationError("rate limited", status_code=429, provider="fake")
                return "ok [1]"

        flaky = Flaky()
        code, out, _ = self.run_cli(["chat"], generator=flaky,
                                    inputs=["drone landing", "sourdough bread", "", "jupiter", "exit"])
        self.assertEqual(code, 0)
        self.assertEqual(len(flaky.prompts), 3)          # empty line skipped
        self.assertIn("Error: rate limited", out)        # session continued after the failure
        self.assertEqual(out.count("Answer:"), 2)

    def test_missing_api_key_gives_friendly_error(self):
        self.write_docs()
        code, _, err = self.run_cli(["ask", "drone"], key=None)
        self.assertEqual(code, 1)
        self.assertIn("GEMINI_API_KEY", err)

    def test_search_works_without_api_key(self):
        self.write_docs()
        self.run_cli(["ingest"])
        with mock.patch.object(cli, "make_generator", side_effect=AssertionError("must not be called")):
            code, out = self.run_cli_no_generator(["search", "drone"])
        self.assertEqual(code, 0)
        self.assertIn("drone.txt", out)

    def run_cli_no_generator(self, argv):
        env = {"DOCS_DIR": str(self.root / "docs"), "INDEX_DIR": str(self.root / "index"),
               "EMBEDDING_MODEL": "fake-hash-embedder"}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(cli, "make_embedder", lambda s: FakeEmbedder()), \
                contextlib.redirect_stdout(out):
            code = cli.main(argv)
        return code, out.getvalue()

    def test_missing_docs_folder_gives_friendly_error(self):
        shutil.rmtree(self.root / "docs")
        code, _, err = self.run_cli(["ingest"])
        self.assertEqual(code, 1)
        self.assertIn("Documents folder not found", err)

    def test_empty_question_gives_friendly_error(self):
        self.write_docs()
        code, _, err = self.run_cli(["search", "   "])
        self.assertEqual(code, 1)
        self.assertIn("empty", err)


if __name__ == "__main__":
    unittest.main()
