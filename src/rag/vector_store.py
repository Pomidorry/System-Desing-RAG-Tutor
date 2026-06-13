from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


_COLLECTION = "system_design"


def build_vector_store(embeddings: Embeddings, persist_dir: str) -> Chroma:
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
