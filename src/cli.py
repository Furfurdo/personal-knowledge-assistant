from __future__ import annotations

import argparse
from pathlib import Path

from config import create_embeddings, create_llm, load_settings
from index_builder import build_faiss_index, iter_knowledge_files, load_documents, split_documents
from qa_engine import answer_question, load_faiss_index


def _add_index_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs-dir", default="knowledge_base", help="Knowledge base directory")
    parser.add_argument("--index-dir", default="vector_store", help="FAISS index directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap")


def _add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", default="vector_store", help="FAISS index directory")
    parser.add_argument("--k", type=int, default=6, help="Top-k retrieval")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved snippets")
    parser.add_argument(
        "--auto-index",
        action="store_true",
        help="Auto build index if missing",
    )
    parser.add_argument("--docs-dir", default="knowledge_base", help="Knowledge base directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for auto-index")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for auto-index",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Knowledge Assistant (RAG CLI)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build vector index from txt/md/pdf files")
    _add_index_options(index_parser)

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question", nargs="?", help="Question text (optional)")
    _add_query_options(ask_parser)

    repl_parser = subparsers.add_parser("repl", help="Interactive Q&A mode")
    _add_query_options(repl_parser)

    web_parser = subparsers.add_parser("web", help="Start local web UI")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=8787, help="Port (default: 8787)")
    web_parser.add_argument("--docs-dir", default="knowledge_base", help="Knowledge base directory")
    web_parser.add_argument("--index-dir", default="vector_store", help="FAISS index directory")
    web_parser.add_argument("--k", type=int, default=6, help="Default top-k retrieval")
    web_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for reindex")
    web_parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for reindex")

    doctor_parser = subparsers.add_parser("doctor", help="Check local setup and readiness")
    doctor_parser.add_argument("--docs-dir", default="knowledge_base", help="Knowledge base directory")
    doctor_parser.add_argument("--index-dir", default="vector_store", help="FAISS index directory")

    return parser


def _index_ready(index_dir: str) -> bool:
    path = Path(index_dir)
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def _build_index_flow(
    settings,
    docs_dir: str,
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    embeddings = create_embeddings(settings)

    print("Loading documents...")
    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        print(f"No txt/md/pdf files found under: {docs_dir}")
        return False
    print(f"Loaded documents: {len(raw_docs)}")

    print("Splitting documents...")
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Chunks: {len(chunks)}")

    print("Building FAISS index...")
    try:
        count = build_faiss_index(chunks, embeddings, index_dir)
    except Exception as exc:
        message = str(exc)
        if "UNIMPLEMENTED" in message or "Error code: 501" in message:
            print("Embedding endpoint is not available on your current provider.")
            print("Fix: set EMBEDDING_PROVIDER=local in .env and run again.")
        else:
            print(f"Failed to build index: {exc}")
        return False

    print(f"Done. Indexed chunks: {count}, saved to: {index_dir}")
    return True


def run_index(args: argparse.Namespace) -> int:
    settings = load_settings()
    ok = _build_index_flow(
        settings=settings,
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    return 0 if ok else 1


def _ensure_index_for_query(args: argparse.Namespace, settings) -> bool:
    if _index_ready(args.index_dir):
        return True

    if not args.auto_index:
        print(f"Index not found in: {args.index_dir}")
        print(
            f"Run this first: python src/cli.py index --docs-dir {args.docs_dir} --index-dir {args.index_dir}"
        )
        print("Or add --auto-index to this command.")
        return False

    print("Index not found. Auto-indexing now...")
    return _build_index_flow(
        settings=settings,
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def run_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not _ensure_index_for_query(args, settings):
        return 1

    question = args.question or input("Question> ").strip()
    if not question:
        print("Empty question.")
        return 1

    embeddings = create_embeddings(settings)
    llm = create_llm(settings)
    index = load_faiss_index(args.index_dir, embeddings)

    try:
        result = answer_question(index, llm, question, k=args.k, include_context=args.show_context)
    except Exception as exc:
        message = str(exc)
        if "dimension" in message.lower() or "assert" in message.lower():
            print("Index/embedding mismatch detected.")
            print("Please rebuild index: python src/cli.py index")
        elif "connection error" in message.lower():
            print("Query failed: network or endpoint is unreachable.")
            print("Check LLM_BASE_URL / API key, then retry.")
        else:
            print(f"Query failed: {exc}")
        return 1
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src}")

    if args.show_context:
        contexts = result.get("contexts", [])
        if contexts:
            print("\nRetrieved Context:")
            for item in contexts:
                print(f"{item['ref']} {item['source']}")
                print(f"  {item['snippet']}")
    return 0


def run_repl(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not _ensure_index_for_query(args, settings):
        return 1

    embeddings = create_embeddings(settings)
    llm = create_llm(settings)
    index = load_faiss_index(args.index_dir, embeddings)

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            result = answer_question(index, llm, question, k=args.k, include_context=args.show_context)
        except Exception as exc:
            message = str(exc)
            if "dimension" in message.lower() or "assert" in message.lower():
                print("Index/embedding mismatch detected. Rebuild index first: python src/cli.py index")
            elif "connection error" in message.lower():
                print("Query failed: network or endpoint is unreachable. Check LLM_BASE_URL / API key.")
            else:
                print(f"Query failed: {exc}")
            continue
        print("\nAssistant>")
        print(result["answer"])

        if result["sources"]:
            print("Sources:")
            for src in result["sources"]:
                print(f"- {src}")

        if args.show_context:
            contexts = result.get("contexts", [])
            if contexts:
                print("Retrieved Context:")
                for item in contexts:
                    print(f"{item['ref']} {item['source']}")
                    print(f"  {item['snippet']}")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    ok = True

    env_exists = Path(".env").exists()
    print(f"[{'OK' if env_exists else 'WARN'}] .env exists: {env_exists}")
    if not env_exists:
        print("      Create it: copy .env.example .env")
        ok = False

    try:
        settings = load_settings()
        print("[OK] API/model config loaded")
        print(f"      LLM model: {settings.llm_model}")
        print(f"      Embedding provider: {settings.embedding_provider}")
        print(f"      Embedding model: {settings.embedding_model}")
    except Exception as exc:
        print(f"[FAIL] Config error: {exc}")
        ok = False

    docs = list(iter_knowledge_files(args.docs_dir))
    print(f"[{'OK' if docs else 'WARN'}] Knowledge files found: {len(docs)}")
    if not docs:
        print(f"      Put .txt/.md/.pdf files under: {args.docs_dir}")
        ok = False

    index_ready = _index_ready(args.index_dir)
    print(f"[{'OK' if index_ready else 'WARN'}] Index ready: {index_ready}")
    if not index_ready:
        print(f"      Build index: python src/cli.py index --docs-dir {args.docs_dir}")
        ok = False

    if ok:
        print("\nSystem is ready.")
    else:
        print("\nSystem is not fully ready. Fix WARN/FAIL items above.")
    return 0 if ok else 1


def run_web(args: argparse.Namespace) -> int:
    try:
        from web_app import create_app
    except ImportError as exc:
        print(f"Web UI dependencies missing: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    app = create_app(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        default_k=args.k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Web UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        return run_index(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "repl":
        return run_repl(args)
    if args.command == "web":
        return run_web(args)
    return run_doctor(args)


if __name__ == "__main__":
    raise SystemExit(main())
