from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


def build_retriever(vector_store: Chroma, k: int = 5) -> VectorStoreRetriever:
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
