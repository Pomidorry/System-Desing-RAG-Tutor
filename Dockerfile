FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling in CUDA (~900 MB saved)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so first startup is fast
RUN python -W ignore -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', \
model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})"

COPY src/ src/
# Knowledge base is NOT baked into the image — users mount their own at runtime
RUN mkdir -p data/knowledge_base

EXPOSE 8501

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
