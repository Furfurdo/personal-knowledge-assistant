from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores import FAISS


def load_faiss_index(index_dir: str, embeddings) -> FAISS:
    path = Path(index_dir)
    if not path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def retrieve_context(index: FAISS, question: str, k: int = 4):
    retriever = index.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def _source_label(metadata: dict) -> str:
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    if page is not None:
        return f"{source}#page={page}"
    return source


def answer_question(
    index: FAISS,
    llm,
    question: str,
    k: int = 4,
    include_context: bool = False,
) -> Dict[str, object]:
    docs = retrieve_context(index, question, k=k)
    if not docs:
        return {
            "answer": "No related context found. Try another question or add more documents.",
            "sources": [],
            "contexts": [],
        }

    context_blocks: List[str] = []
    sources: List[str] = []
    contexts: List[Dict[str, str]] = []

    for i, doc in enumerate(docs, start=1):
        label = _source_label(doc.metadata)
        text = doc.page_content.strip().replace("\n", " ")
        snippet = text[:900]
        context_blocks.append(f"[{i}] source={label}\ncontent={snippet}")
        sources.append(label)
        contexts.append({"ref": f"[{i}]", "source": label, "snippet": snippet})

    prompt = (
        "You are a personal knowledge base assistant.\n"
        "Answer using only the provided context. Do not invent facts.\n"
        "Try your best to answer from relevant context first.\n"
        "Only when none of the context is relevant, say exactly: '\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u8db3\u591f\u4fe1\u606f'.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{chr(10).join(context_blocks)}\n\n"
        "Output in Chinese with:\n"
        "1) concise answer\n"
        "2) evidence references like [1], [2]\n"
        "3) if uncertain, state uncertainty explicitly"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    unique_sources = []
    seen = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        unique_sources.append(source)

    result: Dict[str, object] = {
        "answer": answer.strip(),
        "sources": unique_sources,
        "contexts": [],
    }
    if include_context:
        result["contexts"] = contexts
    return result
