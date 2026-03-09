from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from index_builder import iter_knowledge_files, load_documents, split_documents


def test_iter_and_load_documents(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("# title\nbeta", encoding="utf-8")
    (tmp_path / "ignore.png").write_text("x", encoding="utf-8")

    files = list(iter_knowledge_files(str(tmp_path)))
    assert {f.suffix for f in files} == {".txt", ".md"}

    docs = load_documents(str(tmp_path))
    assert len(docs) == 2
    assert all("source" in d.metadata for d in docs)


def test_split_documents_returns_multiple_chunks() -> None:
    long_text = "rag " * 1200
    docs = [Document(page_content=long_text, metadata={"source": "x"})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
