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
from src.tutor.quiz import generate_quiz, grade, Quiz

app = Flask(__name__, template_folder="templates")

_llm = None
_retriever = None
_quizzes: dict[str, Quiz] = {}


def get_rag():
    global _llm, _retriever
    if _retriever is None:
        s = get_settings()
        vs = build_vector_store(get_embeddings(), s.chroma_persist_dir)
        _retriever = build_retriever(vs)
        _llm = build_llm()
    return _llm, _retriever


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/quiz/start", methods=["POST"])
def quiz_start():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid request body"}), 400
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "empty topic"}), 400
    try:
        llm, retriever = get_rag()
        quiz = generate_quiz(llm, retriever, topic)
    except Exception as exc:
        return jsonify({"error": f"could not generate quiz: {exc}"}), 500

    _quizzes[quiz.id] = quiz
    questions = [
        {"id": idx, "question": q.question, "options": q.options}
        for idx, q in enumerate(quiz.questions)
    ]
    return jsonify({"quiz_id": quiz.id, "topic": quiz.topic, "questions": questions})


@app.route("/quiz/submit", methods=["POST"])
def quiz_submit():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid request body"}), 400
    quiz_id = data.get("quiz_id")
    answers = data.get("answers") or {}
    if not isinstance(answers, dict):
        return jsonify({"error": "invalid answers"}), 400
    quiz = _quizzes.get(quiz_id)
    if quiz is None:
        return jsonify({"error": "quiz not found — please restart"}), 404
    result = grade(quiz, {str(k): v for k, v in answers.items()})
    return jsonify(result)


if __name__ == "__main__":
    get_rag()
    app.run(host="0.0.0.0", port=8501, debug=False)
