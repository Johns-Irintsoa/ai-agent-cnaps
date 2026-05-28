# Fonctionnalités en cours — ai-agent-cnaps

## Pipeline RAG principal — `POST /ask`

Orchestré par `src/inference/service.py::ask_question()`.

### Étapes du pipeline

```
User Query → POST /ask
  │
  ├─ 0. Cache sémantique (Redis)        → HIT : retour immédiat
  │
  ├─ 1. Embedding query (BGE-M3)        → vecteur réutilisé pour ChromaDB
  │
  ├─ 2. Retrieval ChromaDB              → top-8 par similarité vectorielle
  │
  ├─ 3. Retrieval BM25                  → top-8 par score lexical (BM25Okapi)
  │
  ├─ 4. RRF Fusion                      → fusion + déduplication par chunk_id
  │
  ├─ 5. Reranking cosinus               → top-3 documents sélectionnés
  │
  └─ 6. Génération LLM (Mistral 7B)    → réponse française, persona "Lucy"
```

### Détail des étapes

| Étape | Fichier | Description |
|---|---|---|
| Cache sémantique | `src/inference/cache/semantic_cache.py` | Redis + RediSearch HNSW, seuil cosinus 0.92, TTL 24h |
| Embedding | `src/inference/query_retriever.py` | BGE-M3 via Model Runner, LRU cache in-memory (taille 100) |
| ChromaDB retrieval | `src/inference/query_retriever.py` | Similarité vectorielle, k=8 |
| BM25 retrieval | `src/inference/bm25_retriever.py` | BM25Okapi, index construit au démarrage, thread-safe |
| RRF Fusion | `src/inference/multi_query_retriever.py::reciprocal_rank_fusion()` | Combine les deux listes, déduplique, k=60 |
| Reranking | `src/inference/reranking.py` | Cosinus numpy, top-3 |
| Génération | `src/inference/prompting.py::generate_answer()` | Prompt strict : français, CNaPS only, max 3 phrases |

### Réponse retournée

```json
{
  "answer": "...",
  "from_cache": false,
  "evaluation": {
    "timing": {
      "Embedding": 0.312,
      "ChromaDB retrieval": 0.145,
      "BM25": 0.021,
      "RRF Fusion": 0.003,
      "Reranking": 0.089,
      "Generation LLM": 4.521,
      "total": 5.091
    },
    "tokens": {
      "prompt_tokens": 512,
      "completion_tokens": 128,
      "total_tokens": 640
    }
  }
}
```

---

## Ingestion de documents

### Upload PDF — `POST /ingest-pdf`

**Fichiers :** `src/api/app.py`, `src/ingestion/transform/service.py`

Pipeline déclenché à chaque upload :

1. Validation extension `.pdf`
2. Sauvegarde temporaire dans `temp_uploads/`
3. **Parsing** via Docling (`src/ingestion/transform/parsing.py`) → export Markdown
4. **Chunking** Markdown par headers H1/H2/H3 (`src/ingestion/transform/splitting.py`)
5. **Embedding + stockage** dans ChromaDB (`src/ingestion/transform/embedding.py`)
6. **Refresh BM25** en arrière-plan après indexation
7. Nettoyage du fichier temporaire

### Chargement web CNaPS — `POST /ingestion/load/web-data`

**Fichier :** `src/ingestion/load/Service.py`

- Lit les URLs depuis `cnaps_urls.json`
- Scrape le HTML via `UnstructuredLoader`
- Indexe les documents dans ChromaDB

### Classification de documents — `POST /ingestion/filter`

**Fichier :** `src/ingestion/filter/functions.py`

- Classifie les documents d'un répertoire via LLM
- Catégories : `FORMULAIRE`, `TABLEAU`, `TEXTE`, `AUTRE`
- Retourne les documents acceptés et rejetés

---

## Infrastructure transverse

### Index BM25

**Fichier :** `src/inference/bm25_retriever.py`

- Construit automatiquement au démarrage de l'application (lifespan FastAPI)
- Reconstruit après chaque ingestion PDF (`bm25_index.refresh()`)
- Thread-safe via `threading.Lock`
- Retourne `[]` si ChromaDB est vide (pas de crash)

### Cache sémantique Redis

**Fichier :** `src/inference/cache/semantic_cache.py`

- Index RediSearch HNSW créé automatiquement à la connexion
- Clé : hash de la question, préfixe `semcache:`
- Retry automatique avec backoff exponentiel (3 tentatives)
- Cache LRU in-memory pour les embeddings (évite les re-encodages)
- Écriture en fire-and-forget (`asyncio.create_task`)

### Timer RAG

**Fichier :** `src/inference/timer.py`

- Mesure chaque étape du pipeline en secondes et en pourcentage
- Expose `measure()` (synchrone) et `ameasure()` (asynchrone)
- Rapport loggé + retourné dans le champ `evaluation.timing` de la réponse

### Visualisation ChromaDB

**Fichier :** `src/VectorDB/visualize.py`

```bash
# Debug sans LLM : comptage, chunks signature, recherche full-text
python -m src.VectorDB.visualize

# Recherche sémantique avec scores
python -m src.VectorDB.visualize verify "votre question"
python -m src.VectorDB.visualize verify "votre question" 10
```

---

## Endpoints API actifs

| Méthode | Endpoint | Statut | Description |
|---|---|---|---|
| POST | `/ask` | **Actif** | Pipeline RAG complet |
| POST | `/ingest-pdf` | **Actif** | Upload + indexation PDF |
| POST | `/ingestion/load/web-data` | **Actif** | Chargement URLs CNaPS |
| POST | `/ingestion/filter` | **Actif** | Classification LLM de documents |
| GET | `/docs` | **Actif** | Swagger UI |

### Endpoints legacy (non maintenus)

| Endpoint | Problème |
|---|---|
| `POST /scraper/index` | `load_page()` non importé → erreur runtime |
| `POST /scraper/ask` | `answer_question()` non importé → erreur runtime |
| `POST /ingestion/scrap` | Import relatif cassé hors Docker |
| `POST /ingestion/scrap/v1` | Import relatif cassé hors Docker |
| `POST /ingestion/load/test` | Import relatif cassé hors Docker |
| `POST /load/pdfs` | Import relatif cassé hors Docker |

---

## Fonctions implémentées mais non utilisées

| Fonction | Fichier | Note |
|---|---|---|
| `multi_query_retriever_async()` | `src/inference/multi_query_retriever.py` | Remplacée par ChromaDB + BM25 dans `service.py` |
| `generate_answer_multi_query()` | `src/inference/prompting.py` | Définie, jamais appelée |

---

## Stack technique active

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Mistral 7B via Model Runner (OpenAI-compatible) |
| Embeddings | BGE-M3 via Model Runner |
| Vector store | ChromaDB (SQLite backend) |
| Recherche lexicale | BM25Okapi (`rank_bm25`) |
| Cache sémantique | Redis + RediSearch |
| Parsing PDF | Docling |
| Async | `asyncio` + `asyncio.to_thread()` pour les appels bloquants |
