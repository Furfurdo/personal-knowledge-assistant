from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # Fallback for older envs
    from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


@dataclass
class Settings:
    api_key: str
    base_url: str | None
    llm_model: str
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dim
            sign = -1.0 if digest[0] & 1 else 1.0
            vec[index] += sign

        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def load_settings() -> Settings:
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing LLM_API_KEY (or OPENAI_API_KEY).")

    base_url = os.getenv("LLM_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip()
    llm_model = os.getenv("LLM_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        embedding_model = (
            "hash-384"
            if embedding_provider in {"hash", "local"}
            else ("sentence-transformers/all-MiniLM-L6-v2" if embedding_provider == "hf" else "text-embedding-3-small")
        )

    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip() or api_key
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip() or base_url

    return Settings(
        api_key=api_key,
        base_url=base_url or None,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key or None,
        embedding_base_url=embedding_base_url or None,
    )


def create_embeddings(settings: Settings) -> Any:
    if settings.embedding_provider in {"hash", "local"}:
        return HashEmbeddings(dim=384)
    if settings.embedding_provider == "hf":
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError as exc:
            raise ImportError(
                "Local embedding dependencies missing. Run: pip install -r requirements.txt"
            ) from exc
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
    raise ValueError("Unsupported EMBEDDING_PROVIDER. Use 'hash' (or 'local'), 'hf', or 'openai'.")


def create_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.2,
    )
