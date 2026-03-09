from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def iter_knowledge_files(docs_dir: str) -> Iterable[Path]:
    root = Path(docs_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf_documents(path: Path) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    docs: List[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_num},
            )
        )
    return docs


def load_documents(docs_dir: str) -> List[Document]:
    docs: List[Document] = []
    for path in iter_knowledge_files(docs_dir):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            docs.extend(_load_pdf_documents(path))
            continue

        content = _read_text_file(path).strip()
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={"source": str(path)},
            )
        )
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_faiss_index(
    documents: List[Document],
    embeddings,
    index_dir: str,
) -> int:
    if not documents:
        raise ValueError("No documents found to index.")

    vector_store = FAISS.from_documents(documents, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_dir)
    return len(documents)
