from __future__ import annotations

import math

from config import HashEmbeddings, load_settings


def test_hash_embeddings_shape_and_norm() -> None:
    emb = HashEmbeddings(dim=64)
    vec = emb.embed_query("RAG retrieval generation")
    assert len(vec) == 64
    norm = math.sqrt(sum(x * x for x in vec))
    assert 0.99 <= norm <= 1.01


def test_load_settings_local_alias_to_hash(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    settings = load_settings()
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "hash-384"
