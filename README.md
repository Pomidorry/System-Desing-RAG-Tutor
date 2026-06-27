# System Design RAG Tutor

An interactive tutoring chatbot for system design, powered by a RAG pipeline backed by DeepSeek LLM. Ask a question — the system retrieves relevant chunks from your knowledge base and generates a focused, expert-level answer.

## Stack

| Component | Technology |
|---|---|
| LLM | DeepSeek (`deepseek-chat`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) |
| Vector DB | ChromaDB (persistent) |
| RAG | LangChain `ConversationalRetrievalChain` + MMR |
| UI | Flask (topic-based system design quiz) |

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

# 4. Launch UI (Flask dev server)
python src/ui/app.py
```

Open http://localhost:8501

### Docker (run with your own books and API key)

The Docker image ships **without** any knowledge-base documents — you supply your
own books and your own DeepSeek API key. Follow these steps:

**1. Install Docker.** Get [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(Windows/macOS) or Docker Engine + the Compose plugin (Linux). Verify:

```bash
docker --version
docker compose version
```

**2. Get the project** and open a terminal in its root folder:

```bash
git clone <repo-url>
cd System-Desing-RAG-Tutor
```

**3. Add your DeepSeek API key.** Copy the example env file and edit it:

```bash
cp .env.example .env
```

Open `.env` and set your key (get one at https://platform.deepseek.com):

```
DEEPSEEK_API_KEY=sk-your-real-key-here
```

The other variables already have working defaults — leave them as-is.

**4. Add your own books.** Drop your documents into `data/knowledge_base/`.
Supported formats: `.pdf`, `.md`, `.txt`.

```bash
# example
cp ~/Downloads/my-system-design-book.pdf data/knowledge_base/
```

This folder is mounted into the container at runtime, so your files never get
baked into the image.

**5. Build and start:**

```bash
docker compose up --build
```

On the **first** run the container automatically ingests everything in
`data/knowledge_base/` into the vector store before the web server starts
(this can take a few minutes depending on book size). The index is saved to a
named Docker volume (`chroma_data`), so later runs start instantly and skip
re-ingestion.

**6. Open the app:** http://localhost:8501

**7. Stop it:** press `Ctrl+C`, or from another terminal:

```bash
docker compose down
```

#### Updating your books later

Ingestion is skipped whenever the vector store already contains data. After you
add or remove books, reset the index so it gets rebuilt on the next start:

```bash
docker compose down -v   # -v removes the chroma_data volume (the index)
docker compose up        # re-ingests your current books, then starts the app
```

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

No documents are bundled with the project — you provide your own knowledge base.
For Docker, see step 4 above and the "Updating your books later" note.

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
