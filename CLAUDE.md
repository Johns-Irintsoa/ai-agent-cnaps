# CLAUDE.md — ai-agent-cnaps

## 1. Présentation du projet

Système RAG (Retrieval-Augmented Generation) pour CNaPS (caisse de retraite malgache) permettant la recherche sémantique et le Q&A sur des documents PDF et pages web via un LLM local.

---

## 2. Stack technique

| Composant | Rôle |
|---|---|
| **FastAPI** (Python 3.11) | API REST, port 8000 |
| **ChromaDB** (`chromadb/chroma:1.5.x`) | Stockage vectoriel, port 8001 |
| **Redis Stack** | Cache sémantique + RediSearch, port 6379 (réseau interne) |
| **Model Runner** (Docker Desktop) | Inference LLM (Mistral 7B) + embeddings (BGE-M3) + reranker |
| **Angular 19** | Frontend chatbot "Lucy BOT", port 4200 |
| **Nginx** (prod) | Reverse proxy frontend → backend |
| **Docling** | Extraction PDF → Markdown |
| **trafilatura** | Scraping HTML → Markdown |
| **BM25** (`rank_bm25`) | Retrieval lexical hybride |

---

## 3. Architecture des dossiers

```
.
├── src/
│   ├── api/            # Routes FastAPI (app.py, schemas.py)
│   ├── inference/      # Pipeline RAG : retrieval, reranking, LLM, cache
│   │   └── cache/      # Cache sémantique Redis
│   ├── ingestion/
│   │   ├── load/       # Chargement PDF/HTML via UnstructuredLoader
│   │   ├── transform/  # Parsing → Chunking → Embedding
│   │   ├── filter/     # Classification LLM des documents
│   │   ├── scraping/   # Modèles + scrapper web CNaPS
│   │   └── store/      # Écriture dans ChromaDB
│   ├── models/         # Wrappers LLM, embeddings, reranker
│   ├── db/             # Clients Redis et ChromaDB
│   └── config_env.py   # Redirection cache HF/RapidOCR (importé en premier)
├── frontend/           # Angular 19 (chatbot widget Lucy BOT)
├── tests/              # pytest : unit + integration
├── cnaps_urls.json     # URLs CNaPS à ingérer
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 4. Commandes essentielles

### Démarrer la stack complète
```bash
docker compose up --build        # premier lancement
docker compose up                # lancements suivants
```

### Relancer un service spécifique
```bash
docker compose restart backend
docker compose up --build backend   # après modification de code
docker compose logs -f backend      # suivre les logs
```

### Lancer les tests
```bash
# Dans le conteneur backend
docker compose exec backend pytest tests/ -v
docker compose exec backend pytest tests/ -v -m "not integration"  # sans réseau
```

### Appels API principaux
```bash
# Question RAG
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Quels sont les droits à la retraite ?"}'

# Ingestion PDF
curl -X POST http://localhost:8000/ingest-pdf \
  -F "file=@/chemin/vers/document.pdf"

# Ingestion pages web CNaPS
curl -X POST http://localhost:8000/ingestion/web-pages

# Interface swagger
open http://localhost:8000/docs

# Frontend chatbot
open http://localhost:4200
```

---

## 5. Conventions de code

**Nommage**
- `snake_case` pour fonctions et variables, `PascalCase` pour classes
- Préfixe `_` pour fonctions internes (`_pdf_docling`, `_parse_html`, `_fragments_to_markdown`)
- Modules nommés par rôle : `parsing.py`, `splitting.py`, `embedding.py`, `service.py`

**Structure des modules**
- Un `service.py` par domaine orchestre les fonctions du module
- Séparation stricte : load / transform / store / inference
- `logging.getLogger(__name__)` dans chaque module, jamais `print()` en production

**Async/await**
- Toute l'inférence (`ask_question`) est `async`
- Les appels bloquants (ChromaDB, BM25, LLM) passent par `asyncio.to_thread()`
- Cache Redis entièrement async via `aioredis`

**Gestion d'erreurs**
- Chaque route FastAPI a un `try/except` → `HTTPException(status_code=500)`
- Les fonctions de service retournent `None` en cas d'échec, pas d'exception propagée
- `logger.warning` ou `logger.error` systématique avant tout `return None`

**Types**
- Type hints systématiques (`Optional[str]`, `List[Document]`, `Dict[str, Any]`)
- Modèles Pydantic pour toutes les requêtes et réponses API

---

## 6. Points d'attention

**Ordre de démarrage (critique)**
- `redis-cache` doit être `healthy` avant `backend` (healthcheck compose)
- `chromadb` doit être démarré avant `backend`
- Le BM25 index est construit au démarrage (`lifespan`) depuis ChromaDB — ChromaDB vide = index vide

**Variables d'environnement obligatoires**
```
LLM_BASE_URL        # URL du Model Runner (LLM + embeddings + reranker)
LLM_MODEL           # identifiant modèle Mistral
EMBEDDINGS_MODEL    # identifiant modèle BGE-M3
REDIS_PASSWORD      # mot de passe Redis (doit correspondre dans le compose)
COLLECTION_NAME     # collection ChromaDB (défaut : rag_cnaps)
CHROMA_HOST         # "chromadb" en Docker, vide en dev local
```

**Fichiers à ne pas modifier sans précaution**
- `src/config_env.py` : redirige le cache HF avant tout import ML — doit rester le premier import
- `cnaps_urls.json` : structure stricte attendue par `convert_json_to_list()`
- `docker-compose.yml` : le service backend s'appelle `backend` (proxy Angular pointe dessus)

**Proxy frontend**
- En dev Docker : `proxy.conf.json` route `/api` → `http://backend:8000` (nom de service Docker)
- Le préfixe `/api` est supprimé avant d'atteindre le backend (`pathRewrite`)

**Principe clean code**
- Fonctions courtes, une seule responsabilité par fonction
- Pas de logique métier dans les routes FastAPI (déléguer aux services)
- Aucune valeur hardcodée : tout passe par `.env` via `os.getenv()`
- `try/except` obligatoire dans chaque endpoint et chaque appel réseau externe
