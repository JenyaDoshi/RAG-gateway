"""Stage 7 - Application entry point (command line).

    python main.py ingest            build / rebuild the index from data/docs
    python main.py search "..."      show the retrieved chunks only (no LLM, no API key)
    python main.py ask "..."         retrieve + generate one answer
    python main.py chat              ask many questions (default if no command given)
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag.config import Settings, load_settings
from rag.embedder import FastEmbedEmbedder
from rag.errors import GenerationError, IndexNotFoundError, RAGError
from rag.llm import GeminiGenerator
from rag.pipeline import Answer, RAGPipeline
from rag.vector_store import SearchHit


# ---- wiring (tests replace these two functions with fakes) ----------------

def make_embedder(settings: Settings):
    return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)


def make_generator(settings: Settings):
    return GeminiGenerator(
        api_key=settings.require_api_key(),
        model=settings.gemini_model,
        timeout_s=settings.llm_timeout_s,
    )


def build_pipeline(settings: Settings, with_llm: bool) -> RAGPipeline:
    generator = make_generator(settings) if with_llm else None  # fails early if the key is missing
    return RAGPipeline(settings, make_embedder(settings), generator)


# ---- printing ----------------------------------------------------------------

def print_hits(hits: list[SearchHit], full: bool) -> None:
    if not hits:
        print("Retrieved passages: none")
        return
    print(f"Retrieved passages (top {len(hits)}; higher score = more similar):")
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(f"  [{n}] score {hit.score:.3f}  {c.source} (chunk {c.index})")
        body = c.text if full else c.text.replace("\n", " ")
        if not full and len(body) > 160:
            body = body[:160].rstrip() + "..."
        print(textwrap.indent(textwrap.fill(body, 96) if not full else body, "      "))
    print()


def print_answer(answer: Answer) -> None:
    print("Answer:")
    print(textwrap.indent(answer.text, "  "))
    print()
    if answer.cited:
        print("Sources cited:")
        for hit in answer.cited:
            print(f"  - {hit.chunk.source} (chunk {hit.chunk.index})")
    else:
        print("Sources cited: none (the model did not cite any passage)")


# ---- commands --------------------------------------------------------------

def ensure_index(pipeline: RAGPipeline) -> None:
    """Load the saved index; build it first if this is the very first run."""
    try:
        pipeline.load_index()
    except IndexNotFoundError:
        print("No index yet - building it from your documents first.")
        cmd_ingest_with(pipeline)
    if pipeline.is_index_stale():
        print("Note: the documents folder changed since the index was built. "
              "Run 'python main.py ingest' to refresh it.\n")


def cmd_ingest_with(pipeline: RAGPipeline) -> None:
    print(f"Reading documents from {pipeline.settings.docs_dir} ...")
    print("(the first run downloads the embedding model, which can take a minute)")
    report = pipeline.ingest()
    for warning in report.warnings:
        print(f"  warning: {warning}")
    print(f"Indexed {report.documents} document(s) as {report.chunks} chunk(s) "
          f"using {report.embedding_model}.")
    print(f"Index saved to {pipeline.settings.index_dir}\n")


def cmd_ingest(args, settings: Settings) -> int:
    cmd_ingest_with(build_pipeline(settings, with_llm=False))
    return 0


def cmd_search(args, settings: Settings) -> int:
    pipeline = build_pipeline(settings, with_llm=False)
    ensure_index(pipeline)
    print_hits(pipeline.retrieve(args.question, args.top_k), args.full_chunks)
    return 0


def cmd_ask(args, settings: Settings) -> int:
    pipeline = build_pipeline(settings, with_llm=True)
    ensure_index(pipeline)
    hits = pipeline.retrieve(args.question, args.top_k)
    print_hits(hits, args.full_chunks)           # show retrieval BEFORE calling the LLM
    print_answer(pipeline.generate_answer(args.question, hits))
    return 0


def cmd_chat(args, settings: Settings) -> int:
    pipeline = build_pipeline(settings, with_llm=True)
    ensure_index(pipeline)
    print(f"Index ready ({len(pipeline.store.chunks)} chunks). "
          f"Model: {settings.gemini_model}. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            hits = pipeline.retrieve(question, args.top_k)
            print()
            print_hits(hits, args.full_chunks)
            print_answer(pipeline.generate_answer(question, hits))
        except RAGError as exc:                  # a failed question must not end the session
            print(f"\nError: {exc}")
    return 0


# ---- argument parsing ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--top-k", type=int, default=None, help="chunks to retrieve (default: TOP_K, 4)")
    shared.add_argument("--full-chunks", action="store_true", help="print full retrieved text, not a preview")

    parser = argparse.ArgumentParser(description="Document Q&A with retrieval-augmented generation.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("ingest", help="build the index from data/docs")
    p_search = sub.add_parser("search", parents=[shared], help="show retrieved chunks only")
    p_search.add_argument("question")
    p_ask = sub.add_parser("ask", parents=[shared], help="ask one question")
    p_ask.add_argument("question")
    sub.add_parser("chat", parents=[shared], help="interactive question loop")
    return parser


COMMANDS = {"ingest": cmd_ingest, "search": cmd_search, "ask": cmd_ask, "chat": cmd_chat}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # avoid Windows console crashes

    raw = list(sys.argv[1:] if argv is None else argv) or ["chat"]
    args = build_parser().parse_args(raw)
    try:
        return COMMANDS[args.command](args, load_settings())
    except ValueError as exc:                    # e.g. an empty question
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(f"\nLLM error: {exc}", file=sys.stderr)
        return 1
    except RAGError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
