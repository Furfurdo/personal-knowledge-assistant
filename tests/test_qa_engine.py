from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from qa_engine import _source_label, answer_question


class _Retriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _question):
        return self._docs


class _Index:
    def __init__(self, docs):
        self._docs = docs

    def as_retriever(self, **_kwargs):
        return _Retriever(self._docs)


@dataclass
class _Resp:
    content: str


class _LLM:
    def invoke(self, _prompt):
        return _Resp(content="这是一个测试回答 [1]")


def test_source_label_with_page() -> None:
    label = _source_label({"source": "notes/a.pdf", "page": 3})
    assert label == "notes/a.pdf#page=3"


def test_answer_question_with_context() -> None:
    docs = [
        Document(page_content="rag is retrieval augmented generation", metadata={"source": "a.md"}),
        Document(page_content="faiss is a vector index", metadata={"source": "a.md"}),
    ]
    result = answer_question(_Index(docs), _LLM(), "what is rag", k=4, include_context=True)
    assert "测试回答" in result["answer"]
    assert result["sources"] == ["a.md"]
    assert len(result["contexts"]) == 2


def test_answer_question_when_empty_docs() -> None:
    result = answer_question(_Index([]), _LLM(), "q", k=4, include_context=True)
    assert "No related context found" in result["answer"]
    assert result["sources"] == []
