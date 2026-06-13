# System Design RAG Tutor

An interactive tutoring chatbot for system design, powered by a RAG pipeline backed by DeepSeek LLM. Ask a question — the system retrieves relevant chunks from your knowledge base and generates a focused, expert-level answer.

## Stack

| Component | Technology |
|---|---|
| LLM | DeepSeek (`deepseek-chat`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) |
| Vector DB | ChromaDB (persistent) |
| RAG | LangChain `ConversationalRetrievalChain` + MMR |
| UI | Streamlit |

## Quick Start

### Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY

# 3. Ingest documents into the vector store (run once)
python -m src.ingestion.loader

# 4. Launch UI
streamlit run src/ui/app.py
```

Open http://localhost:8501

### Docker

```bash
docker-compose up --build
```

On first run, ingestion happens automatically before the UI starts. The vector store is persisted in a Docker volume — subsequent runs skip re-ingestion.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key | — (required) |
| `DEEPSEEK_BASE_URL` | API base URL | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | Model name | `deepseek-chat` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `.chroma` |
| `KNOWLEDGE_BASE_DIR` | Knowledge base directory | `data/knowledge_base` |

## Knowledge Base

Drop `.md`, `.txt`, or `.pdf` files into `data/knowledge_base/` and re-run ingestion:

```bash
python -m src.ingestion.loader
```

The default setup includes 5 system design books (Kleppmann, Alex Xu, Petrov, Burns).

## Tests

```bash
pytest -v
```

## Architecture

```
User question
      │
      ▼
 Streamlit UI
      │
      ▼
ConversationalRetrievalChain
      │                │
      ▼                ▼
 DeepSeek LLM    MMR Retriever (k=5)
 deepseek-chat        │
                       ▼
                  ChromaDB
                (BGE embeddings)
                       │
                       ▼
             data/knowledge_base/
```

## Project Structure

```
src/
  config.py              # Pydantic-settings config
  ingestion/
    loader.py            # Load .md/.txt/.pdf, batched ingestion
    chunker.py           # RecursiveCharacterTextSplitter
  rag/
    embeddings.py        # HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5)
    vector_store.py      # ChromaDB persistent client
    retriever.py         # MMR retriever
  llm/
    deepseek_client.py   # ChatOpenAI pointed at DeepSeek
  tutor/
    chain.py             # ConversationalRetrievalChain + ask()
  ui/
    app.py               # Streamlit chat with history and source attribution
data/
  knowledge_base/        # Place your documents here
tests/                   # pytest, all external APIs mocked
```
