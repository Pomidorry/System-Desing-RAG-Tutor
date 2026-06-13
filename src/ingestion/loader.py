import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents


_LOADERS = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
}


def load_knowledge_base(directory: str) -> list[Document]:
    docs: list[Document] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _LOADERS:
                continue
            path = os.path.join(root, fname)
            loader = _LOADERS[ext](path)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path
            docs.extend(loaded)
    return docs


_BATCH_SIZE = 500


def ingest(knowledge_base_dir: str, vector_store) -> int:
    docs = load_knowledge_base(knowledge_base_dir)
    if not docs:
        return 0
    chunks = chunk_documents(docs)
    for i in range(0, len(chunks), _BATCH_SIZE):
        vector_store.add_documents(chunks[i : i + _BATCH_SIZE])
    return len(chunks)


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from src.config import get_settings
    from src.rag.vector_store import build_vector_store
    from src.rag.embeddings import get_embeddings

    s = get_settings()
    vs = build_vector_store(get_embeddings(), s.chroma_persist_dir)
    count = ingest(s.knowledge_base_dir, vs)
    print(f"Ingested {count} chunks.")
