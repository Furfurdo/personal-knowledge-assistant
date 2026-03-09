# Personal Knowledge Assistant

Turn your notes into a searchable, answerable knowledge base.

This project provides a local RAG workflow for personal documents:

- upload or place files (`.txt`, `.md`, `.pdf`)
- build a local FAISS index
- ask questions in CLI or Web UI
- get answers with source references

## Why this project

Personal notes usually become hard to search over time.  
This tool keeps everything local and gives you practical question-answer access to your own materials.

## Features

- Local-first retrieval pipeline (FAISS)
- Document support: `txt`, `md`, `pdf`
- CLI commands: `doctor`, `index`, `ask`, `repl`, `web`
- Web UI with:
  - upload
  - auto-reindex option
  - answer evidence snippets
  - one-click wipe (double confirmation)
- Offline-safe embedding mode (`hash`) for restricted networks

## Architecture (MVP)

1. Ingestion
  - scan files under `knowledge_base/`
  - parse text/pdf
  - chunk documents
2. Indexing
  - create embeddings
  - build local FAISS index in `vector_store/`
3. QA
  - retrieve top-k chunks
  - generate answer with citations

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

Recommended `.env`:

```env
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=hash
EMBEDDING_MODEL=hash-384
```

### 3) Add files

Put your files in:

```text
knowledge_base/
```

### 4) Verify setup

```bash
python src/cli.py doctor
```

### 5) Use

CLI:

```bash
python src/cli.py ask "What does my note say about RAG?" --auto-index --show-context
```

Web:

```bash
python src/cli.py web
```

Open: `http://127.0.0.1:8787`

## CLI Reference

- `python src/cli.py doctor`
- `python src/cli.py index`
- `python src/cli.py ask "your question" --show-context`
- `python src/cli.py repl`
- `python src/cli.py web --port 8787`

## Quality Signals

- CI on push and pull requests
- Linting with Ruff
- Unit tests with Pytest
- Reproducible environment via `requirements.txt`

## Project Structure

```text
src/
  cli.py
  config.py
  index_builder.py
  qa_engine.py
  web_app.py
tests/
knowledge_base/
vector_store/
```

## Security & Privacy

- `.env` is ignored by git
- local knowledge files are ignored by default
- vector index is ignored by default
- keep your API keys private

## Roadmap

- Better ranking and query rewriting
- Optional OCR for scanned PDFs
- Multi-knowledge-base profiles
- Exportable Q&A history
