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
