# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ai-agent-cnaps** is a modular RAG (Retrieval-Augmented Generation) system for CNaPS (Madagascar's pension system). It enables semantic document search and question-answering over structured/unstructured documents using local LLM inference (no external API keys required).

### Core Architecture

The system follows a pipeline architecture:

```
User Query -> FastAPI (/ask)
  |
  v
  Semantic Cache Check (Redis)
  |
  v
  Multi-Query Retrieval (query reformulation + parallel vector search)
  |
  v
  RRF Fusion (reciprocal rank fusion)
  |
  v
  Reranking (cosine similarity)
  |
  v
  LLM Answer Generation
  |
  v
  Response + Cache Storage
```

**Main Entry Point:** `src/api/app.py` (FastAPI app) → `src/inference/service.py::ask_question()` (async orchestrator)

### Key Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **Inference** | RAG pipeline orchestration & retrieval | service.py, multi_query_retriever.py, query_retriever.py, reranking.py, prompting.py |
| **Ingestion** | Document loading, parsing, chunking, classification | load/, transform/, filter/, chunck/chuncking.py |
| **Models** | LLM & embedding wrappers (OpenAI-compatible API) | models/llm.py, models/embedding.py, models/reranking.py |
| **Caching** | Redis-backed semantic cache with RediSearch | inference/cache/semantic_cache.py, db/redis_client.py |
| **VectorDB** | Chroma initialization & persistence | VectorDB/initialize.py |
| **Classification** | AI-based document type classification | data_classificateur/ |

### External Infrastructure

- **Model Runner** (Docker) - Runs local LLMs (Mistral 7B) via OpenAI-compatible API
- **Chroma** - Persistent vector store (SQLite backend at `./vector_cnaps_db`)
- **Redis** - Semantic caching + RediSearch index

## Build & Run Commands

### Docker Compose (Recommended)

```bash
# First run (downloads models ~5 GB for Mistral + BGE-M3)
docker compose up --build

# Subsequent runs (instant startup from cached volumes)
docker compose up
```

**Services started:**
- FastAPI app on http://localhost:8000
- Redis on internal network (no external port)
- Model Runner (assumed running separately or via compose extension)

**Health check:**
- Swagger UI: http://localhost:8000/docs
- POST `/ask` to test the RAG pipeline

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env: set LLM_BASE_URL to your Model Runner instance

# Run FastAPI directly
uvicorn src.api.app:app --reload --port 8000
```

**Prerequisites:**
- Model Runner must be running (external Docker container or service)
- Redis must be accessible at `redis://redis-cache:6379/0` (set `REDIS_URL` in `.env`)
- Vector DB directory must exist: `./vector_cnaps_db`

### Testing Endpoints

```bash
# Query the RAG system
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelle est l'"'"'article 5?"}'

# View API schema
curl http://localhost:8000/docs
```

## Configuration

### Environment Variables (.env)

**LLM & Embeddings:**
```env
LLM_BASE_URL=http://model-runner.docker.internal:12434/engines/llama.cpp/v1
LLM_MODEL=huggingface.co/bartowski/mistral-7b-instruct-v0.3-gguf:q4_k_m
LLM_API_KEY=no-key
EMBEDDINGS_MODEL=huggingface.co/gpustack/bge-m3-gguf
LLM_RERANKING_BASE_URL=http://model-runner.docker.internal:12434/engines/llama.cpp/v1
LLM_RERANKING_MODEL=huggingface.co/gpustack/bge-reranker-v2-m3-gguf
```

**Vector Database:**
```env
VECTOR_DB_DIR=./vector_cnaps_db
COLLECTION_NAME=rag_cnaps
```

**Chunking:**
```env
CHUNCKING_MAX_TOKENS=450
CHUNKING_OVERLAP_TOKENS=50
```

**Caching:**
```env
REDIS_URL=redis://redis-cache:6379/0
REDIS_PASSWORD=your_password
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_SECONDS=86400
CACHE_MAX_ANSWER_TOKENS=4000
```

**API:**
```env
API_HOST=0.0.0.0
API_PORT=8000
```

See `.env.example` for all available options.

## Core Workflows

### RAG Query Processing (Main Workflow)

**Flow:** `POST /ask` → `ask_question(user_query)` → Response

1. **Semantic Cache Check** - Check Redis for semantically similar cached queries (threshold: 0.92)
2. **Multi-Query Retrieval** - Generate 3 query reformulations via LLM
3. **Parallel Vector Search** - Search Chroma for all 4 queries (original + 3 variants)
4. **RRF Fusion** - Combine & deduplicate results using reciprocal rank fusion
5. **Reranking** - Score documents via cosine similarity
6. **Answer Generation** - Inject top documents into prompt template + invoke LLM
7. **Cache Storage** - Store answer in Redis asynchronously

**Key functions:**
- `src/inference/service.py::ask_question()` - Main async orchestrator
- `src/inference/multi_query_retriever.py::multi_query_retriever_async()` - Query generation + fusion
- `src/inference/prompting.py::generate_answer_multi_query()` - LLM answer generation

### Document Ingestion (PDF Upload)

**Flow:** `POST /ingest-pdf` → `transform_pdf()` → Chroma indexing

1. **Parsing** - Extract text/tables from PDF via Docling (with PyTesseract OCR fallback)
2. **Chunking** - Split by document type (HTML headers, Markdown, tabular rows)
3. **Embedding** - Generate vectors via BGE-M3
4. **Storage** - Persist to Chroma with metadata

**Key functions:**
- `src/ingestion/transform/service.py::transform_pdf()` - Orchestrator
- `src/ingestion/transform/parsing.py::_pdf_docling()` - Docling extraction
- `src/ingestion/transform/splitting.py::chuncking_md_data()` - Smart chunking
- `src/ingestion/transform/embedding.py::embed_chunks()` - Vectorization

### Web Data Ingestion

**Flow:** `POST /ingestion/load/web-data` → Load URLs from `cnaps_urls.json` → Chroma

- Reads JSON config defining CNaPS website URLs
- Uses `UnstructuredLoader` to scrape & parse HTML
- Chunks & indexes to Chroma

**Key functions:**
- `src/ingestion/load/Service.py::load_web_data()` - Main orchestrator
- `src/ingestion/load/UnstructuredLoader.py::load_html_from_url()` - Web scraping

### Document Classification

**Purpose:** Filter/categorize ingested documents (`FORMULAIRE`, `TABLEAU`, `TEXTE`, `AUTRE`)

- Uses LLM-based classification to organize documents
- Helps downstream systems handle different document types appropriately

**Key functions:**
- `src/data_classificateur/ClassificationLLM.py::build_prompt()` - Classification prompt
- `src/ingestion/filter/functions.py::process_unstructured_data()` - Filter orchestrator

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ask` | Query documents via RAG (main endpoint) |
| POST | `/ingest-pdf` | Upload & index a single PDF |
| POST | `/ingestion/load/web-data` | Load & index URLs from `cnaps_urls.json` |
| POST | `/ingestion/filter` | Classify documents in a directory |
| POST | `/load/pdfs` | Load PDFs from filesystem path |
| GET | `/docs` | Swagger UI |

Legacy endpoints: `/scraper/index`, `/scraper/ask`, `/ingestion/scrap`

## Code Organization

```
src/
├── api/                    # FastAPI application
│   ├── app.py             # Route handlers & orchestration
│   └── schemas.py         # Pydantic request/response models
├── inference/             # RAG core pipeline
│   ├── service.py         # Main async orchestrator (ask_question)
│   ├── multi_query_retriever.py  # Query reformulation + RRF
│   ├── query_retriever.py # Vector similarity search
│   ├── reranking.py       # Document scoring
│   ├── prompting.py       # Answer generation & prompt templates
│   ├── evaluation.py      # Quality assessment
│   └── cache/
│       ├── semantic_cache.py     # Redis + RediSearch cache
│       └── embedding_cache.py    # LRU cache for embeddings
├── ingestion/             # Document processing pipeline
│   ├── load/              # Web scraping & PDF loading
│   ├── transform/         # Parsing (Docling) -> chunking -> embedding
│   ├── filter/            # Document classification
│   ├── scrap/             # Web scraping utilities
│   ├── chunck/            # Smart chunking by doc type
│   └── store/             # Vector store operations
├── models/                # LLM & embedding wrappers
│   ├── llm.py            # ChatOpenAI wrapper
│   ├── embedding.py      # BGE-M3 embeddings
│   └── reranking.py      # Reranker config
├── db/                    # Database clients
│   └── redis_client.py    # Async Redis with pool & health checks
├── data_classificateur/   # Document type classification
├── VectorDB/              # Chroma initialization
└── config_env.py          # Environment setup (cache redirection)
```

## Important Implementation Details

### Async/Await Pattern
All I/O operations in `src/inference/service.py` use `asyncio`. `semantic_cache` is fully async (RediSearch queries). Blocking vector store calls run in thread executor via `asyncio.to_thread()`.

### Prompt Engineering
The system prompt in `src/inference/prompting.py::PROMPT_TEMPLATE` enforces:
- French-only responses ("Lucy" persona)
- Context-only answers (no hallucination)
- Concise format (max 3 sentences or 5 bullets)

### Vector Search Strategy
- Multi-query generates 3 variations + runs 4 searches in parallel
- RRF (Reciprocal Rank Fusion) merges results, deduplicates by document ID
- Reranking scores final candidates via cosine similarity
- Top-K documents injected into answer generation prompt

### Document Chunking
- **HTML:** Split by H1/H2/H3 headers
- **Markdown:** Split by `#`/`##`/`###`
- **Tabular:** Line-by-line (no overlap)
- **Generic:** 450 tokens max, 50 token overlap
- **Metadata:** Preserves source, page, table flags

### Caching
- **Semantic Cache:** Redis stores (query embedding, answer, metadata)
- **Hit Threshold:** 0.92 cosine similarity (configurable via `CACHE_SIMILARITY_THRESHOLD`)
- **TTL:** 24 hours default
- **LRU Embedding Cache:** In-memory cache for recent embeddings

## Common Development Scenarios

### Adding a New Ingestion Source
1. Create loader in `src/ingestion/load/` or extend `UnstructuredLoader.py`
2. Add endpoint in `src/api/app.py`
3. Documents flow through `transform_pdf()` pipeline (parsing → chunking → embedding)
4. Stored automatically in Chroma via collection

### Modifying LLM Behavior
1. Update prompt in `src/inference/prompting.py::PROMPT_TEMPLATE`
2. Or modify `src/models/llm.py::LLMClient` for temperature/max_tokens

### Adjusting Retrieval Strategy
- **Query variations:** Edit `src/inference/multi_query_retriever.py::MULTI_QUERY_PROMPT`
- **Reranking:** Modify `src/inference/reranking.py::get_reranked_documents(k_final=...)`
- **Top-K documents:** Adjust `top_k_per_query`, `top_k_final` in `src/inference/service.py::ask_question()`

### Debugging Cache Issues
- Check Redis health: `src/db/redis_client.py::RedisClient.healthcheck()`
- Clear semantic cache: Drop `idx:semantic_cache` index in Redis
- Bypass cache: Comment out `await semantic_cache.get()` in `src/inference/service.py`

## Key Dependencies

- `langchain`, `langchain-openai`, `langchain-chroma` - RAG orchestration
- `chromadb` - Vector database
- `redis` - Semantic cache backend
- `docling` - PDF extraction & parsing
- `fastapi`, `uvicorn` - API framework
- `torch`, `transformers` - ML inference support
- `beautifulsoup4`, `pypdf`, `python-docx` - Document parsing
- `pytesseract`, `opencv-python` - OCR support

See `requirements.txt` for the full list.

## Deployment Notes

- **Docker image:** Multi-stage build (builder + runtime), `python:3.11-slim` base
- **Volumes:** `chroma_data` (vector store), `redis_data` (Redis persistence)
- **Networking:** Services communicate via internal Docker bridge (`app-network`); Redis has no external port
- **Model Runner:** Accessible via `model-runner.docker.internal` (requires Docker Desktop or `extra_hosts` config)
- **First run:** Models download ~5.8 GB total, stored in Docker volumes for reuse
