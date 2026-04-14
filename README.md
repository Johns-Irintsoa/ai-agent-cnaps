# ai-agent-cnaps

Architecture modulaire LangChain + Ollama pour un agent IA open source — sans cle API externe.

## Architecture

```
docker-compose network (bridge)
  ┌─────────────────────────┐        ┌──────────────────────────────────────┐
  │  app (python:3.11)      │──────> │  ollama (ollama/ollama)              │
  │  FastAPI sur :8000      │        │  API REST sur :11434                 │
  │  attend Ollama healthy  │        │  mistral-small3.2 + nomic-embed-text │
  └─────────────────────────┘        └──────────────────────────────────────┘
                                               │
                                       volume: ollama_data
                                       (modeles persistes)
```

### Modeles utilises
| Usage | Modele | Taille |
|---|---|---|
| LLM (chat) | `mistral-small3.2` | ~5 GB |
| Embeddings | `nomic-embed-text` | ~274 MB |

## Structure du projet

```
ai-agent-cnaps/
├── src/
│   ├── llm/
│   │   ├── __init__.py          # export OllamaLLM
│   │   └── chat.py              # Wrapper ChatOllama (mistral-small3.2)
│   ├── vector_database/
│   │   ├── __init__.py          # export VectorDatabase
│   │   └── store.py             # OllamaEmbeddings + InMemoryVectorStore
│   ├── api/
│   │   ├── __init__.py          # export app
│   │   └── app.py               # FastAPI — endpoint POST /chat
│   ├── __init__.py
│   └── main.py                  # Entry point uvicorn
├── .env                         # Configuration (gitignore)
├── .gitignore
├── docker-compose.yml           # Orchestration ollama + app
├── Dockerfile                   # Image Python 3.11
├── requirements.txt             # Dependances Python
└── README.md
```

## Demarrage du projet

### Prerequis

- [Docker](https://www.docker.com/get-started) et Docker Compose

### Installation

**1. Cloner le depot**
```bash
git clone https://github.com/votre-username/ai-agent-cnaps.git
cd ai-agent-cnaps
```

**2. Verifier la configuration dans `.env`**
```env
# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=mistral:7b-instruct-q4_K_M
EMBEDDINGS_MODEL=nomic-embed-text

# API
API_HOST=0.0.0.0
API_PORT=8000
```
> Tous les modeles et l'URL Ollama sont configurables via `.env` — aucune cle API requise.

### Lancer avec Docker

**Premier lancement (telechargement des modeles ~5 GB) :**
```bash
docker compose up --build
```

**Relancer sans rebuild :**
```bash
docker compose up
```

> Le premier demarrage telecharge `mistral-small3.2` (~5 GB) et `nomic-embed-text` (~274 MB).
> Les lancements suivants sont instantanes grace au volume `ollama_data`.

## Utilisation de l'API

### Endpoint `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bonjour, qui es-tu ?"}'
```

**Reponse :**
```json
{
  "response": "Je suis un assistant IA..."
}
```

### Documentation interactive (Swagger UI)

Disponible sur `http://localhost:8000/docs` apres le demarrage.

## Modules

### `src/llm/`
Encapsule `ChatOllama`. La classe `OllamaLLM` expose une methode `invoke(message)` qui retourne une chaine de caracteres.

### `src/vector_database/`
Encapsule `OllamaEmbeddings` + `InMemoryVectorStore`. La classe `VectorDatabase` expose :
- `add_documents(texts)` — indexe une liste de textes
- `similarity_search(query, k)` — retourne les `k` documents les plus proches

### `src/api/`
Application FastAPI. Le chatbot est appele cote backend — le client envoie uniquement un message texte.
