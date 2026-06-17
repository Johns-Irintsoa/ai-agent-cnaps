# SQL Agent — Dernière Période de Cotisation CNaPS

**Date :** 2026-06-17  
**Branche :** admin/backend  
**Statut :** Approuvé

---

## Context

Le projet RAG CNaPS répond aux questions réglementaires via ChromaDB. Il manque la capacité de répondre aux questions personnelles des affiliés (ex : "Quelle est ma dernière période de cotisation ?"), qui nécessitent un accès à la base Oracle de production (SIG.CIT2).

On ajoute un SQL Agent LangChain branché sur Oracle, intégré de façon transparente dans le endpoint `/ask` existant via un routing stateless basé sur la détection d'intention + de matricule.

---

## Architecture

```
POST /ask
     │
     ▼
src/inference/service.py
     │
     ├─ _is_cotisation_intent(message) → True, _extract_matricule() → None
     │     └─ RAGResponse { answer: "Veuillez saisir votre matricule.", needs_matricule: True }
     │
     ├─ _extract_matricule(message) → "512196"
     │     └─ SQL Agent pipeline → Oracle → RAGResponse { answer: "...", needs_matricule: False }
     │
     └─ ni intent ni matricule
           └─ RAG pipeline existant (inchangé)
```

**Nouveaux fichiers :**

| Fichier | Rôle |
|---|---|
| `src/db/oracle_client.py` | Singleton OracleClient : SQLAlchemy engine + LangChain SQLDatabase |
| `src/inference/SQLAgent/__init__.py` | Package marker |
| `src/inference/SQLAgent/agent_toolkit.py` | Initialise SQLDatabaseToolkit (4 outils) |
| `src/inference/SQLAgent/SQL_agent.py` | Construit l'agent ReAct + `run_sql_agent()` async |

**Fichiers modifiés :**

| Fichier | Modification |
|---|---|
| `src/inference/service.py` | Ajout routing + appel `run_sql_agent()` |
| `src/api/schemas.py` | Ajout `needs_matricule: bool = False` dans `RAGResponse` |
| `requirements.txt` | Ajout `oracledb`, `SQLAlchemy`, `langchain-community` |
| `.env.example` | Ajout bloc `ORACLE_*` |

---

## Data Flow

### Tour 1 — Intention détectée, pas de matricule

```
POST /ask  {"message": "Quelle est ma dernière période de cotisation ?"}

service.py :
  _is_cotisation_intent(message) → True
    (mots-clés : période, cotisation, cotisé, dernière, dernier, cotisations)
  _extract_matricule(message) → None

→ RAGResponse {
    answer: "Pour consulter votre dernière période de cotisation, veuillez me fournir votre numéro de matricule.",
    needs_matricule: True,
    metadata: None
  }
```

### Tour 2 — Matricule détecté, SQL Agent s'exécute

```
POST /ask  {"message": "512196"}   (ou "Mon matricule est 512196")

service.py :
  _extract_matricule(message) → "512196"   (regex : \b\d{5,7}\b)

SQL Agent (ReAct, max_iterations=5) :
  Iter 1 : list_tables         → identifie SIG.CIT2
  Iter 2 : schema(SIG.CIT2)    → lit colonnes (CIT_PERIODE, TRAVAILLEUR_MATRICULE, ...)
  Iter 3 : query_checker       → valide SELECT MAX(CIT_PERIODE) FROM SIG.CIT2 WHERE ...
  Iter 4 : query               → exécute, retourne "202412"
  Iter 5 : raisonnement final  → formule réponse en français

→ RAGResponse {
    answer: "La dernière période de cotisation du matricule 512196 est décembre 2024 (202412).",
    needs_matricule: False,
    metadata: None
  }
```

### Fallback — Aucun intent ni matricule

```
POST /ask  {"message": "Quels sont les droits à la retraite ?"}
→ RAG pipeline existant (inchangé)
```

---

## Interfaces

### `src/db/oracle_client.py`

Pattern identique à `ChromaClient` (singleton + retry + backoff).

```python
class OracleClient:
    _engine: Optional[Engine] = None
    _db: Optional[SQLDatabase] = None

    @classmethod
    def get_engine(cls) -> Engine:
        """SQLAlchemy engine, thin mode oracledb. Singleton + retry x3."""

    @classmethod
    def get_db(cls) -> SQLDatabase:
        """LangChain SQLDatabase limité au schéma ORACLE_SCHEMA."""

    @classmethod
    def healthcheck(cls) -> bool: ...
```

Connection string :
```
oracle+oracledb://USER:PASS@HOST:PORT/SID
```
Exemple : `oracle+oracledb://user:pass@host.docker.internal:1522/xe`

Mode thin (pas d'Oracle Instant Client requis — compatible Docker).

`SQLDatabase.from_uri(conn_str, schema="SIG", include_tables=["CIT2"])` — limite l'agent aux tables autorisées, évite l'exposition de tout le schéma Oracle.

### `src/inference/SQLAgent/agent_toolkit.py`

```python
def get_toolkit(db: SQLDatabase, llm: BaseChatModel) -> SQLDatabaseToolkit:
    """
    Retourne SQLDatabaseToolkit avec les 4 outils :
    sql_db_list_tables, sql_db_schema, sql_db_query_checker, sql_db_query.
    """
```

### `src/inference/SQLAgent/SQL_agent.py`

```python
def build_sql_agent(llm: BaseChatModel, toolkit: SQLDatabaseToolkit) -> AgentExecutor:
    """
    create_sql_agent avec system prompt incluant :
    - Réponse en français
    - Si question hors périmètre DB → répondre 'HORS_PERIMETRE'
    - max_iterations=5, handle_parsing_errors=True
    """

async def run_sql_agent(question: str) -> Dict[str, Any]:
    """
    Entrée : question en langage naturel (contient le matricule)
    Sortie : {"answer": str, "needs_matricule": False, "metadata": None}
    Gestion : timeout, HORS_PERIMETRE, erreurs Oracle → logger.error, réponse de fallback
    """
```

### Routing dans `src/inference/service.py`

```python
_COTISATION_KEYWORDS = {"période", "cotisation", "cotisé", "cotisations", "dernier", "dernière"}

def _is_cotisation_intent(message: str) -> bool:
    words = set(message.lower().split())
    return bool(words & _COTISATION_KEYWORDS)

def _extract_matricule(message: str) -> Optional[str]:
    match = re.search(r'\b(\d{5,7})\b', message)
    return match.group(1) if match else None
```

Modification de `ask_question()` :

```python
async def ask_question(user_query: str) -> Dict[str, Any]:
    matricule = _extract_matricule(user_query)
    if matricule:
        return await run_sql_agent(user_query)

    if _is_cotisation_intent(user_query):
        return {
            "answer": "Pour consulter votre dernière période de cotisation, veuillez me fournir votre numéro de matricule.",
            "needs_matricule": True,
            "metadata": None,
            "evaluation": {},
            "from_cache": False,
        }

    # ... pipeline RAG existant (inchangé)
```

---

## Variables d'environnement

Ajout dans `.env.example` :

```env
# Oracle Database
ORACLE_HOST=localhost           # host.docker.internal en Docker
ORACLE_PORT=1522
ORACLE_SID=xe
ORACLE_USER=
ORACLE_PASSWORD=
ORACLE_SCHEMA=SIG              # schéma exposé à l'agent (limite la surface de données)
```

---

## Dépendances

Ajout dans `requirements.txt` :

```
oracledb>=2.0.0
SQLAlchemy>=2.0.0
langchain-community>=0.2.0
```

`langchain` et `langchain-openai` sont déjà présents dans le projet.

---

## Gestion d'erreurs

| Situation | Comportement |
|---|---|
| Oracle inaccessible (démarrage) | `logger.warning`, `OracleClient._db = None`, pas de crash au démarrage |
| Oracle inaccessible (requête) | `logger.error`, réponse: "Service de données indisponible." |
| Agent dépasse `max_iterations` | `logger.warning`, réponse: "Je n'ai pas pu répondre à votre question." |
| Agent répond `HORS_PERIMETRE` | réponse: "Cette information n'est pas disponible dans la base de données." |
| Matricule inexistant (résultat NULL) | réponse: "Aucune période de cotisation trouvée pour ce matricule." |

---

## Requête Oracle cible

```sql
SELECT MAX(CIT_PERIODE) AS DERNIERE_PERIODE
FROM SIG.CIT2
WHERE TRAVAILLEUR_MATRICULE = :matricule
```

L'agent génère cette requête de façon autonome après introspection du schéma.

---

## Vérification

1. **Démarrage** — `docker compose up backend` → logs sans erreur Oracle (warning si DB absente)
2. **Tour 1** — `POST /ask {"message": "Quelle est ma dernière période de cotisation ?"}` → `needs_matricule: true`
3. **Tour 2** — `POST /ask {"message": "512196"}` → réponse avec la période
4. **RAG inchangé** — `POST /ask {"message": "Quels sont les droits à la retraite ?"}` → réponse RAG normale
5. **Oracle KO** — Couper Oracle → réponse gracieuse sans crash
6. **Matricule inconnu** — `POST /ask {"message": "000001"}` → "Aucune période trouvée..."
