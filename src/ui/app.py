import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.config import get_settings
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import build_vector_store
from src.rag.retriever import build_retriever
from src.llm.deepseek_client import build_llm
from src.tutor.chain import build_chain, ask


st.set_page_config(page_title="System Design Tutor", page_icon="🏗️", layout="wide")
st.title("🏗️ System Design Tutor")
st.caption("Ask me anything about distributed systems, databases, caching, and more.")


@st.cache_resource
def _load_chain():
    s = get_settings()
    vs = build_vector_store(get_embeddings(), s.chroma_persist_dir)
    retriever = build_retriever(vs)
    llm = build_llm()
    return build_chain(llm=llm, retriever=retriever)


chain = _load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a system design question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(chain, prompt, st.session_state.history)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- `{s}`")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append((prompt, answer))
