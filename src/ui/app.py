import sys
import warnings
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify
from src.config import get_settings
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import build_vector_store
from src.rag.retriever import build_retriever
from src.llm.deepseek_client import build_llm
from src.tutor.chain import build_chain, ask as chain_ask

app = Flask(__name__, template_folder="templates")

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        s = get_settings()
        vs = build_vector_store(get_embeddings(), s.chroma_persist_dir)
        retriever = build_retriever(vs)
        llm = build_llm()
        _chain = build_chain(llm=llm, retriever=retriever)
    return _chain


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    history = [tuple(pair) for pair in data.get("history", [])]
    if not question:
        return jsonify({"error": "empty question"}), 400
    result = chain_ask(get_chain(), question, history)
    return jsonify({"answer": result["answer"], "sources": result["sources"]})


if __name__ == "__main__":
    get_chain()
    app.run(host="0.0.0.0", port=8501, debug=False)
