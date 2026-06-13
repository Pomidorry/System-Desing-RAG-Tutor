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


from src.ingestion.loader import load_knowledge_base


def test_load_knowledge_base_loads_markdown(tmp_path):
    md_file = tmp_path / "intro.md"
    md_file.write_text("# System Design\nLoad balancing distributes traffic.", encoding="utf-8")
    docs = load_knowledge_base(str(tmp_path))
    assert len(docs) == 1
    assert "load balancing" in docs[0].page_content.lower()
    assert docs[0].metadata["source"].endswith("intro.md")


def test_load_knowledge_base_empty_dir(tmp_path):
    docs = load_knowledge_base(str(tmp_path))
    assert docs == []
