from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import PromptTemplate


_SYSTEM_PROMPT = PromptTemplate.from_template(
    """You are an expert system design tutor. Use the retrieved context to answer the question.
Be concise, accurate, and pedagogical. If the context doesn't cover the question, say so clearly.

Context:
{context}

Question: {question}
Answer:"""
)


def build_chain(llm: BaseChatModel, retriever: BaseRetriever) -> ConversationalRetrievalChain:
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": _SYSTEM_PROMPT},
        return_source_documents=True,
        verbose=False,
    )


def ask(
    chain: ConversationalRetrievalChain,
    question: str,
    history: list[tuple[str, str]],
) -> dict:
    result = chain.invoke({"question": question, "chat_history": history})
    sources = [doc.metadata.get("source", "") for doc in result.get("source_documents", [])]
    return {"answer": result["answer"], "sources": list(set(sources))}
