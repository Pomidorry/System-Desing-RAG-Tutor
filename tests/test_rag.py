import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_get_embeddings_returns_huggingface():
    from src.rag.embeddings import get_embeddings
    emb = get_embeddings()
    assert hasattr(emb, "embed_query")


def test_build_vector_store_creates_collection(tmp_path):
    from src.rag.embeddings import get_embeddings
    from src.rag.vector_store import build_vector_store

    emb = get_embeddings()
    vs = build_vector_store(emb, str(tmp_path / "chroma"))
    assert vs is not None


def test_vector_store_add_and_search(tmp_path):
    from src.rag.embeddings import get_embeddings
    from src.rag.vector_store import build_vector_store

    emb = get_embeddings()
    vs = build_vector_store(emb, str(tmp_path / "chroma"))
    doc = Document(page_content="Load balancers distribute HTTP traffic", metadata={"source": "lb.md"})
    vs.add_documents([doc])
    results = vs.similarity_search("traffic distribution", k=1)
    assert len(results) == 1
    assert "traffic" in results[0].page_content.lower()
