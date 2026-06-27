#!/bin/sh
set -e

# Run ingestion only if Chroma collection is empty
python -W ignore -c "
import sys
try:
    import chromadb
    client = chromadb.PersistentClient(path='${CHROMA_PERSIST_DIR:-.chroma}')
    col = client.get_collection('system_design')
    count = col.count()
    sys.exit(0 if count > 0 else 1)
except Exception:
    sys.exit(1)
" && echo "Vector store already populated, skipping ingestion." || {
    echo "Ingesting knowledge base..."
    python -W ignore -m src.ingestion.loader
}

# 1 worker is mandatory: quizzes and the RAG pipeline live in process memory.
# Threads allow concurrent requests sharing that state; long timeout covers slow LLM calls.
exec gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:8501 src.ui.app:app
