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


def test_build_retriever_returns_retriever(tmp_path):
    from src.rag.embeddings import get_embeddings
    from src.rag.vector_store import build_vector_store
    from src.rag.retriever import build_retriever
    from langchain_core.documents import Document

    emb = get_embeddings()
    vs = build_vector_store(emb, str(tmp_path / "chroma"))
    vs.add_documents([
        Document(page_content="CDN caches static content close to users", metadata={"source": "cdn.md"}),
        Document(page_content="Load balancer routes requests to servers", metadata={"source": "lb.md"}),
    ])
    retriever = build_retriever(vs, k=2)
    results = retriever.invoke("how to cache content")
    assert len(results) >= 1
