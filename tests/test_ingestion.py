from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents


def test_chunk_documents_splits_long_text():
    docs = [Document(page_content="word " * 500, metadata={"source": "test.md"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="Hello world", metadata={"source": "foo.md"})]
    chunks = chunk_documents(docs)
    assert all(c.metadata["source"] == "foo.md" for c in chunks)


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []
