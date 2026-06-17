# SQL Agent Cotisation CNaPS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Intégrer un LangChain SQL Agent (Oracle DB) dans le pipeline `/ask` existant pour permettre à Lucy Bot de répondre aux questions de cotisation en détectant l'intention et le matricule de l'affilié.

**Architecture:** Routing stateless dans `service.py` : mots-clés cotisation sans matricule → demande matricule (`needs_matricule=True`) ; matricule détecté → `run_sql_agent()` → Oracle XE via `OracleClient` singleton + `SQLDatabaseToolkit` ReAct (5 iterations max). RAG pipeline inchangé pour toutes les autres questions.

**Tech Stack:** Python 3.11, LangChain Community (SQLDatabaseToolkit, create_sql_agent), oracledb 2.x thin mode, SQLAlchemy 2.x, pytest + unittest.mock

## Global Constraints

- Python 3.11 — type hints systématiques sur toutes les fonctions publiques
- `snake_case` fonctions/variables, `PascalCase` classes, préfixe `_` pour fonctions internes
- `logging.getLogger(__name__)` dans chaque module — jamais `print()`
- Variables d'env lues dans les méthodes (pas au niveau module) pour compatibilité `load_dotenv()`
- `oracledb` en thin mode — zéro dépendance Oracle Instant Client
- `try/except` + `logger.error` + réponse de fallback dans chaque appel réseau
- Connexion Oracle : `host.docker.internal:1522` (Docker) / `localhost:1522` (dev), SID=xe, schéma=SIG
- Tests : `unittest.mock` pour mocker Oracle — aucune connexion réelle en test unitaire
- Commits fréquents après chaque tâche complétée

---

## File Map

| Action | Chemin | Responsabilité |
|---|---|---|
| Modify | `requirements.txt` | Ajouter oracledb, SQLAlchemy |
| Modify | `.env.example` | Ajouter variables ORACLE_* |
| Modify | `src/api/schemas.py` | Ajouter `needs_matricule` dans RAGResponse |
| Create | `src/db/oracle_client.py` | Singleton OracleClient (engine + SQLDatabase) |
| Create | `src/inference/SQLAgent/__init__.py` | Package marker |
| Create | `src/inference/SQLAgent/agent_toolkit.py` | Initialise SQLDatabaseToolkit |
| Create | `src/inference/SQLAgent/SQL_agent.py` | build_sql_agent + run_sql_agent |
| Modify | `src/inference/service.py` | Routing cotisation + appel run_sql_agent |
| Create | `tests/test_oracle_client.py` | Tests OracleClient |
| Create | `tests/test_sql_agent_toolkit.py` | Tests toolkit |
| Create | `tests/test_sql_agent.py` | Tests run_sql_agent |
| Create | `tests/test_service_sql_routing.py` | Tests routing service |

---

## Task 1: Dépendances + Schema + Env

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `src/api/schemas.py`

**Interfaces:**
- Produces: `RAGResponse.needs_matricule: bool = False` (utilisé par Tasks 4 et 5)

- [ ] **Step 1 : Ajouter les dépendances dans `requirements.txt`**

Ajouter après la ligne `langchain-community` :

```
oracledb>=2.0.0
SQLAlchemy>=2.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2 : Ajouter les variables Oracle dans `.env.example`**

Ajouter à la fin du fichier `.env.example` :

```env
# Oracle Database (SQL Agent)
ORACLE_HOST=localhost           # host.docker.internal en Docker
ORACLE_PORT=1522
ORACLE_SID=xe
ORACLE_USER=
ORACLE_PASSWORD=
ORACLE_SCHEMA=SIG              # schéma Oracle exposé à l'agent
```

- [ ] **Step 3 : Ajouter `needs_matricule` dans `src/api/schemas.py`**

Remplacer la classe `RAGResponse` existante (lignes 12-14) :

```python
class RAGResponse(BaseModel):
    answer: str
    metadata: Optional[QueryMetaData] = None
    needs_matricule: bool = False
```

- [ ] **Step 4 : Vérifier l'import oracledb**

```bash
pip install oracledb SQLAlchemy
python -c "import oracledb; import sqlalchemy; print('OK')"
```

Résultat attendu : `OK`

- [ ] **Step 5 : Commit**

```bash
git add requirements.txt .env.example src/api/schemas.py
git commit -m "feat: add Oracle dependencies and needs_matricule schema field"
```

---

## Task 2: OracleClient Singleton

**Files:**
- Create: `src/db/oracle_client.py`
- Create: `tests/test_oracle_client.py`

**Interfaces:**
- Consumes: env vars `ORACLE_HOST`, `ORACLE_PORT`, `ORACLE_SID`, `ORACLE_USER`, `ORACLE_PASSWORD`, `ORACLE_SCHEMA`
- Produces:
  - `OracleClient.get_db() -> Optional[SQLDatabase]` — utilisé par Task 4 (SQL_agent.py)
  - `OracleClient.healthcheck() -> bool`
  - `OracleClient.close() -> None`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_oracle_client.py` :

```python
import pytest
from unittest.mock import patch, MagicMock
from src.db.oracle_client import OracleClient


def setup_function():
    """Réinitialise le singleton avant chaque test."""
    OracleClient._engine = None
    OracleClient._db = None
    OracleClient._initialized = False


def test_get_engine_returns_engine_on_success():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.oracle_client.create_engine", return_value=mock_engine):
        engine = OracleClient.get_engine()

    assert engine is mock_engine


def test_get_engine_returns_none_on_connection_failure():
    with patch("src.db.oracle_client.create_engine", side_effect=Exception("Connection refused")):
        engine = OracleClient.get_engine()

    assert engine is None


def test_get_engine_is_singleton():
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: MagicMock()
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.oracle_client.create_engine", return_value=mock_engine) as mock_create:
        OracleClient.get_engine()
        OracleClient.get_engine()

    mock_create.assert_called_once()


def test_get_db_returns_none_when_engine_unavailable():
    OracleClient._initialized = True
    OracleClient._engine = None

    result = OracleClient.get_db()

    assert result is None


def test_get_db_returns_sqldatabase_when_engine_available():
    mock_engine = MagicMock()
    mock_db = MagicMock()
    OracleClient._initialized = True
    OracleClient._engine = mock_engine

    with patch("src.db.oracle_client.SQLDatabase", return_value=mock_db):
        result = OracleClient.get_db()

    assert result is mock_db


def test_healthcheck_returns_false_when_engine_unavailable():
    OracleClient._initialized = True
    OracleClient._engine = None

    assert OracleClient.healthcheck() is False


def test_healthcheck_returns_true_when_engine_available():
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: MagicMock()
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    OracleClient._initialized = True
    OracleClient._engine = mock_engine

    assert OracleClient.healthcheck() is True


def test_close_resets_singleton():
    OracleClient._engine = MagicMock()
    OracleClient._db = MagicMock()
    OracleClient._initialized = True

    OracleClient.close()

    assert OracleClient._engine is None
    assert OracleClient._db is None
    assert OracleClient._initialized is False
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
pytest tests/test_oracle_client.py -v
```

Résultat attendu : `ModuleNotFoundError: No module named 'src.db.oracle_client'`

- [ ] **Step 3 : Créer `src/db/oracle_client.py`**

```python
"""
Client Oracle DB : SQLAlchemy engine + LangChain SQLDatabase.
Singleton avec tentative unique de connexion (gracieux si Oracle absent).
"""
import logging
import os
import time
from typing import Optional

from sqlalchemy import create_engine, Engine
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)


class OracleClient:
    """Singleton pour la connexion Oracle (SQLAlchemy + LangChain SQLDatabase)."""

    _engine: Optional[Engine] = None
    _db: Optional[SQLDatabase] = None
    _initialized: bool = False

    @classmethod
    def get_engine(cls) -> Optional[Engine]:
        """Retourne le SQLAlchemy engine (tentative unique, None si Oracle inaccessible)."""
        if not cls._initialized:
            cls._initialized = True
            cls._engine = cls._try_connect()
        return cls._engine

    @classmethod
    def get_db(cls) -> Optional[SQLDatabase]:
        """Retourne le LangChain SQLDatabase limité au schéma ORACLE_SCHEMA."""
        if cls._db is None:
            engine = cls.get_engine()
            if engine is None:
                return None
            try:
                schema = os.getenv("ORACLE_SCHEMA", "SIG")
                cls._db = SQLDatabase(engine, schema=schema, include_tables=["CIT2"])
                logger.info("SQLDatabase Oracle initialisé (schéma=%s)", schema)
            except Exception as e:
                logger.error("Impossible d'initialiser SQLDatabase Oracle : %s", e)
        return cls._db

    @classmethod
    def _try_connect(cls) -> Optional[Engine]:
        """Établit la connexion Oracle avec retry x3 et backoff. Retourne None si échec."""
        host = os.getenv("ORACLE_HOST", "localhost")
        port = os.getenv("ORACLE_PORT", "1522")
        sid = os.getenv("ORACLE_SID", "xe")
        user = os.getenv("ORACLE_USER", "")
        password = os.getenv("ORACLE_PASSWORD", "")
        retries = 3
        backoff = 1.0

        conn_str = f"oracle+oracledb://{user}:{password}@{host}:{port}/{sid}"
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                engine = create_engine(conn_str, echo=False)
                with engine.connect():
                    pass
                logger.info("OracleClient connecté à %s:%s/%s", host, port, sid)
                return engine
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Tentative %d/%d échouée pour Oracle : %s. Nouvel essai dans %.1fs",
                    attempt + 1, retries, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        logger.warning(
            "Oracle inaccessible après %d tentatives (%s). Fonctionnalité SQL désactivée.",
            retries, last_exc,
        )
        return None

    @classmethod
    def healthcheck(cls) -> bool:
        """Vérifie que Oracle est accessible."""
        try:
            engine = cls.get_engine()
            if engine is None:
                return False
            with engine.connect():
                pass
            return True
        except Exception:
            return False

    @classmethod
    def close(cls) -> None:
        """Réinitialise le singleton."""
        if cls._engine:
            cls._engine.dispose()
        cls._engine = None
        cls._db = None
        cls._initialized = False
        logger.info("OracleClient réinitialisé")
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
pytest tests/test_oracle_client.py -v
```

Résultat attendu : tous les tests `PASSED`

- [ ] **Step 5 : Commit**

```bash
git add src/db/oracle_client.py tests/test_oracle_client.py
git commit -m "feat: add OracleClient singleton with SQLDatabase and graceful fallback"
```

---

## Task 3: Agent Toolkit

**Files:**
- Create: `src/inference/SQLAgent/__init__.py`
- Create: `src/inference/SQLAgent/agent_toolkit.py`
- Create: `tests/test_sql_agent_toolkit.py`

**Interfaces:**
- Consumes: `OracleClient.get_db()` → `SQLDatabase` ; `LLMClient().model` → `BaseChatModel`
- Produces: `get_toolkit(db, llm) -> SQLDatabaseToolkit` — utilisé par Task 4

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_sql_agent_toolkit.py` :

```python
from unittest.mock import MagicMock
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from src.inference.SQLAgent.agent_toolkit import get_toolkit


def test_get_toolkit_returns_sqldatabasetoolkit():
    mock_db = MagicMock()
    mock_llm = MagicMock()

    toolkit = get_toolkit(mock_db, mock_llm)

    assert isinstance(toolkit, SQLDatabaseToolkit)


def test_toolkit_has_four_required_tools():
    mock_db = MagicMock()
    mock_db.dialect = "oracle"
    mock_db.get_usable_table_names.return_value = ["CIT2"]
    mock_llm = MagicMock()

    toolkit = get_toolkit(mock_db, mock_llm)
    tool_names = {t.name for t in toolkit.get_tools()}

    assert "sql_db_list_tables" in tool_names
    assert "sql_db_schema" in tool_names
    assert "sql_db_query" in tool_names
    assert "sql_db_query_checker" in tool_names


def test_toolkit_receives_correct_db_and_llm():
    mock_db = MagicMock()
    mock_llm = MagicMock()

    toolkit = get_toolkit(mock_db, mock_llm)

    assert toolkit.db is mock_db
    assert toolkit.llm is mock_llm
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
pytest tests/test_sql_agent_toolkit.py -v
```

Résultat attendu : `ModuleNotFoundError: No module named 'src.inference.SQLAgent'`

- [ ] **Step 3 : Créer le package et `agent_toolkit.py`**

Créer `src/inference/SQLAgent/__init__.py` (vide) :
```python
```

Créer `src/inference/SQLAgent/agent_toolkit.py` :

```python
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel


def get_toolkit(db: SQLDatabase, llm: BaseChatModel) -> SQLDatabaseToolkit:
    """Retourne un SQLDatabaseToolkit avec les 4 outils : list_tables, schema, query, query_checker."""
    return SQLDatabaseToolkit(db=db, llm=llm)
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
pytest tests/test_sql_agent_toolkit.py -v
```

Résultat attendu : tous les tests `PASSED`

- [ ] **Step 5 : Commit**

```bash
git add src/inference/SQLAgent/__init__.py src/inference/SQLAgent/agent_toolkit.py tests/test_sql_agent_toolkit.py
git commit -m "feat: add SQLAgent package and agent_toolkit with SQLDatabaseToolkit"
```

---

## Task 4: SQL Agent Core

**Files:**
- Create: `src/inference/SQLAgent/SQL_agent.py`
- Create: `tests/test_sql_agent.py`

**Interfaces:**
- Consumes:
  - `OracleClient.get_db() -> Optional[SQLDatabase]` (Task 2)
  - `get_toolkit(db, llm) -> SQLDatabaseToolkit` (Task 3)
  - `LLMClient().model -> BaseChatModel` (existant : `src/models/llm.py`)
- Produces:
  - `run_sql_agent(question: str) -> Dict[str, Any]` — utilisé par Task 5
  - Dict keys : `answer: str`, `needs_matricule: bool`, `metadata: None`, `evaluation: dict`, `from_cache: bool`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_sql_agent.py` :

```python
import pytest
from unittest.mock import patch, MagicMock

# Import au niveau module — fonctionne car les patches ciblent les attributs du module,
# pas la référence locale à run_sql_agent.
from src.inference.SQLAgent.SQL_agent import run_sql_agent


@pytest.mark.asyncio
async def test_run_sql_agent_returns_answer_when_db_available():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "La dernière période est 202412."}

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("Dernière période matricule 512196")

    assert result["answer"] == "La dernière période est 202412."
    assert result["needs_matricule"] is False
    assert result["metadata"] is None


@pytest.mark.asyncio
async def test_run_sql_agent_returns_unavailable_when_db_is_none():
    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=None):
        result = await run_sql_agent("512196")

    assert "indisponible" in result["answer"].lower()
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_run_sql_agent_handles_hors_perimetre():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "HORS_PERIMETRE"}

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("Quelque chose hors périmètre")

    assert result["answer"] == "Cette information n'est pas disponible dans la base de données CNaPS."
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_run_sql_agent_handles_exception_gracefully():
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = Exception("Oracle timeout")

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("512196")

    assert result["needs_matricule"] is False
    assert len(result["answer"]) > 0
```

> **Note :** `pytest-asyncio` est requis (ajouté en Task 1). Si le test échoue avec `RuntimeError: no running event loop`, vérifier que `pytest-asyncio` est installé et ajouter dans `tests/conftest.py` : `pytest_plugins = ['pytest_asyncio']` ou configurer `asyncio_mode = "auto"` dans `pytest.ini`.

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
pytest tests/test_sql_agent.py -v
```

Résultat attendu : `ModuleNotFoundError: No module named 'src.inference.SQLAgent.SQL_agent'`

- [ ] **Step 3 : Créer `src/inference/SQLAgent/SQL_agent.py`**

```python
"""
SQL Agent LangChain — répond aux questions sur les données Oracle CNaPS.
Utilise SQLDatabaseToolkit (ReAct) avec max 5 itérations.
"""
import asyncio
import logging
from typing import Dict, Any, Optional

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.language_models import BaseChatModel

from src.db.oracle_client import OracleClient
from src.inference.SQLAgent.agent_toolkit import get_toolkit
from src.models.llm import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PREFIX = """Tu es Lucy, l'assistante CNaPS Madagascar. Tu réponds aux affiliés en français.
Utilise les outils SQL pour interroger la base de données Oracle CNaPS.
Si la question n'est pas liée aux données disponibles, réponds exactement : HORS_PERIMETRE
Sois concis et précis. Ne révèle jamais les détails techniques (noms de tables, colonnes brutes)."""

_MSG_UNAVAILABLE = "Le service de données est temporairement indisponible. Veuillez réessayer plus tard."
_MSG_OUT_OF_SCOPE = "Cette information n'est pas disponible dans la base de données CNaPS."
_MSG_TIMEOUT = "Je n'ai pas pu obtenir la réponse. Veuillez reformuler ou contacter le service CNaPS."

_FALLBACK_RESULT: Dict[str, Any] = {
    "needs_matricule": False,
    "metadata": None,
    "evaluation": {},
    "from_cache": False,
}


def build_sql_agent(llm: BaseChatModel, toolkit: SQLDatabaseToolkit) -> AgentExecutor:
    """Construit l'AgentExecutor ReAct avec le toolkit SQL."""
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        prefix=_SYSTEM_PREFIX,
    )


async def run_sql_agent(question: str) -> Dict[str, Any]:
    """
    Exécute le SQL Agent sur une question en langage naturel.
    Retourne un dict compatible avec la structure de retour de ask_question().
    """
    db = OracleClient.get_db()
    if db is None:
        logger.error("Oracle DB indisponible — SQL Agent non exécuté")
        return {**_FALLBACK_RESULT, "answer": _MSG_UNAVAILABLE}

    try:
        llm = LLMClient().model
        toolkit = get_toolkit(db, llm)
        agent = build_sql_agent(llm, toolkit)

        result = await asyncio.to_thread(agent.invoke, {"input": question})
        output: str = result.get("output", "")

        if "HORS_PERIMETRE" in output:
            logger.info("SQL Agent : question hors périmètre")
            return {**_FALLBACK_RESULT, "answer": _MSG_OUT_OF_SCOPE}

        return {**_FALLBACK_RESULT, "answer": output}

    except Exception as e:
        logger.error("Erreur SQL Agent : %s", e)
        return {**_FALLBACK_RESULT, "answer": _MSG_TIMEOUT}
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
pytest tests/test_sql_agent.py -v
```

Résultat attendu : tous les tests `PASSED`

- [ ] **Step 5 : Commit**

```bash
git add src/inference/SQLAgent/SQL_agent.py tests/test_sql_agent.py
git commit -m "feat: add SQL_agent with run_sql_agent, HORS_PERIMETRE handling and graceful fallbacks"
```

---

## Task 5: Routing dans service.py

**Files:**
- Modify: `src/inference/service.py`
- Create: `tests/test_service_sql_routing.py`

**Interfaces:**
- Consumes:
  - `run_sql_agent(question: str) -> Dict[str, Any]` (Task 4)
  - `_is_cotisation_intent(message: str) -> bool` (nouvelle fonction interne)
  - `_extract_matricule(message: str) -> Optional[str]` (nouvelle fonction interne)
- Produces: `ask_question(user_query)` retourne maintenant 3 chemins possibles (SQL agent, demande matricule, RAG)

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_service_sql_routing.py` :

```python
import pytest
from unittest.mock import patch, AsyncMock

from src.inference.service import _is_cotisation_intent, _extract_matricule


# --- Tests de détection d'intention ---

def test_detects_cotisation_keyword():
    assert _is_cotisation_intent("Quelle est ma dernière période de cotisation ?") is True

def test_detects_cotise_keyword():
    assert _is_cotisation_intent("J'ai cotisé pendant combien de temps ?") is True

def test_detects_periode_keyword():
    assert _is_cotisation_intent("Quelle est ma dernière période ?") is True

def test_no_intent_for_unrelated_question():
    assert _is_cotisation_intent("Quels sont les droits à la retraite ?") is False

def test_no_intent_for_generic_question():
    assert _is_cotisation_intent("Comment contacter la CNaPS ?") is False


# --- Tests d'extraction de matricule ---

def test_extracts_matricule_alone():
    assert _extract_matricule("512196") == "512196"

def test_extracts_matricule_in_sentence():
    assert _extract_matricule("Mon matricule est 512196 merci") == "512196"

def test_extracts_7_digit_matricule():
    assert _extract_matricule("1234567") == "1234567"

def test_extracts_5_digit_matricule():
    assert _extract_matricule("12345") == "12345"

def test_no_matricule_for_8_digits():
    assert _extract_matricule("12345678") is None

def test_no_matricule_for_4_digits():
    assert _extract_matricule("1234") is None

def test_no_matricule_in_text_question():
    assert _extract_matricule("Quelle est ma dernière période de cotisation ?") is None


# --- Tests de routing dans ask_question ---

@pytest.mark.asyncio
async def test_ask_question_routes_to_sql_agent_when_matricule_present():
    mock_result = {
        "answer": "Dernière période : 202412",
        "needs_matricule": False,
        "metadata": None,
        "evaluation": {},
        "from_cache": False,
    }
    with patch("src.inference.service.run_sql_agent", new_callable=AsyncMock, return_value=mock_result):
        from src.inference.service import ask_question
        result = await ask_question("512196")

    assert result["answer"] == "Dernière période : 202412"
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_ask_question_returns_needs_matricule_when_intent_no_matricule():
    with patch("src.inference.service.run_sql_agent") as mock_sql:
        from src.inference.service import ask_question
        result = await ask_question("Quelle est ma dernière période de cotisation ?")

    mock_sql.assert_not_called()
    assert result["needs_matricule"] is True
    assert "matricule" in result["answer"].lower()
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
pytest tests/test_service_sql_routing.py -v
```

Résultat attendu : `ImportError` ou `AttributeError` sur `_is_cotisation_intent`

- [ ] **Step 3 : Modifier `src/inference/service.py`**

Ajouter les imports en tête de fichier (après les imports existants) :

```python
import re
from typing import Optional
from .SQLAgent.SQL_agent import run_sql_agent
```

Ajouter les deux fonctions de routing **avant** `ask_question()` :

```python
_COTISATION_KEYWORDS = {"période", "cotisation", "cotisé", "cotisations", "dernière", "dernier"}

def _is_cotisation_intent(message: str) -> bool:
    words = set(message.lower().split())
    return bool(words & _COTISATION_KEYWORDS)

def _extract_matricule(message: str) -> Optional[str]:
    match = re.search(r'\b(\d{5,7})\b', message)
    return match.group(1) if match else None
```

Modifier `ask_question()` — ajouter le bloc de routing **au début de la fonction**, avant le timer et le cache :

```python
async def ask_question(user_query: str) -> Dict[str, Any]:
    # Routing SQL Agent (stateless)
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

    timer = RAGTimer()
    # ... reste du pipeline RAG existant inchangé
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
pytest tests/test_service_sql_routing.py -v
```

Résultat attendu : tous les tests `PASSED`

- [ ] **Step 5 : Vérifier que les tests RAG existants sont toujours verts**

```bash
pytest tests/ -v -m "not integration"
```

Résultat attendu : aucune régression sur les tests existants

- [ ] **Step 6 : Commit**

```bash
git add src/inference/service.py tests/test_service_sql_routing.py
git commit -m "feat: add SQL Agent routing in ask_question (cotisation intent + matricule detection)"
```

---

## Vérification End-to-End

Une fois le backend lancé (`docker compose up backend`) :

```bash
# 1. Tour 1 — intent sans matricule → needs_matricule: true
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelle est ma dernière période de cotisation ?"}'
# Attendu: {"answer": "...matricule...", "needs_matricule": true}

# 2. Tour 2 — matricule → SQL Agent
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "512196"}'
# Attendu: {"answer": "La dernière période est ...", "needs_matricule": false}

# 3. RAG inchangé
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Quels sont les droits à la retraite ?"}'
# Attendu: réponse RAG normale, needs_matricule: false

# 4. Oracle KO — couper Oracle, relancer
# Attendu: {"answer": "...indisponible...", "needs_matricule": false} — pas de crash

# 5. Matricule inconnu
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "000001"}'
# Attendu: réponse agent indiquant aucune période trouvée
```
