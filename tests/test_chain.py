import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.retrievers import BaseRetriever


class _FakeChatModel(BaseChatModel):
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Load balancers improve availability."))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake"


class _FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return [Document(
            page_content="A load balancer distributes incoming traffic across multiple servers.",
            metadata={"source": "lb.md"},
        )]


def test_build_chain_returns_chain():
    from src.tutor.chain import build_chain
    chain = build_chain(llm=_FakeChatModel(), retriever=_FakeRetriever())
    assert chain is not None


def test_chain_ask_returns_answer_and_sources():
    from src.tutor.chain import build_chain, ask

    chain = build_chain(llm=_FakeChatModel(), retriever=_FakeRetriever())
    result = ask(chain, "What is a load balancer?", history=[])
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
