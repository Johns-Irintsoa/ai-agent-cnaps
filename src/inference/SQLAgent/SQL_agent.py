"""
SQL Agent — répond aux questions sur les données Oracle CNaPS.
Pipeline : Dynamic Few-Shot + SQLDatabaseToolkit + create_sql_agent.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_community.agent_toolkits import create_sql_agent
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from src.db.oracle_client import OracleClient
from src.models.embedding import embedding_manager
from src.models.llm import LLMClient

logger = logging.getLogger(__name__)

# ── System prompt statique (prefix du few-shot) ───────────────────────────────

_SYSTEM_PREFIX = """You are an agent designed to interact with a Oracle 11g SQL database.
Given an input question, create a syntactically correct Oracle 11g query to run,
then look at the results and return the answer in French.
Always limit results to at most 5 rows unless the user specifies otherwise.

You MUST double check your query before executing it.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
Use ROWNUM for row limiting — FETCH FIRST is NOT supported in Oracle 11g.
Always start by listing tables, then query the schema of relevant tables.

Tu es Lucy, l'assistante CNaPS Madagascar. Reponds en francais.
Si la question n'est pas liee aux donnees disponibles, reponds exactement : HORS_PERIMETRE
Ne revele jamais les noms de colonnes ou tables bruts dans ta reponse finale.

Voici des exemples de requetes Oracle 11g similaires a la demande :"""

_MSG_UNAVAILABLE = "Le service de données est temporairement indisponible. Veuillez réessayer plus tard."
_MSG_OUT_OF_SCOPE = "Cette information n'est pas disponible dans la base de données CNaPS."
_MSG_TIMEOUT = "Je n'ai pas pu obtenir la réponse. Veuillez reformuler ou contacter le service CNaPS."

_FALLBACK_RESULT: Dict[str, Any] = {
    "needs_auth": False,
    "needs_matricule": False,
    "metadata": None,
    "evaluation": {},
    "from_cache": False,
}

# ── Dynamic Few-Shot — sélecteur sémantique (singleton) ──────────────────────

_FEW_SHOT_SELECTOR: Optional[SemanticSimilarityExampleSelector] = None

_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "query"],
    template="Question : {input}\nRequête Oracle 11g : {query}",
)


def _get_few_shot_selector() -> Optional[SemanticSimilarityExampleSelector]:
    global _FEW_SHOT_SELECTOR
    if _FEW_SHOT_SELECTOR is not None:
        return _FEW_SHOT_SELECTOR
    try:
        lib_path = Path(__file__).parent / "lib_dynamic_few_shot.json"
        examples = json.loads(lib_path.read_text(encoding="utf-8"))
        _FEW_SHOT_SELECTOR = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,
            embeddings=embedding_manager.model,
            vectorstore_cls=Chroma,
            k=2,
            input_keys=["input"],
        )
        logger.info("FewShotSelector initialise avec %d exemples", len(examples))
    except Exception as e:
        logger.warning("FewShotSelector non disponible : %s", e)
    return _FEW_SHOT_SELECTOR


# ── Construction du prompt ────────────────────────────────────────────────────

def _build_agent_prompt(
    selector: Optional[SemanticSimilarityExampleSelector],
) -> ChatPromptTemplate:
    """
    Construit le ChatPromptTemplate complet.
    Si le sélecteur est disponible, le FewShotPromptTemplate injecte automatiquement
    les 2 exemples les plus pertinents au moment du formatage (via {input}).
    """
    if selector is not None:
        system_prompt = FewShotPromptTemplate(
            example_selector=selector,
            example_prompt=_EXAMPLE_PROMPT,
            prefix=_SYSTEM_PREFIX,
            suffix="",
            input_variables=["input"],
        )
        system_msg = SystemMessagePromptTemplate(prompt=system_prompt)
    else:
        system_msg = SystemMessagePromptTemplate.from_template(_SYSTEM_PREFIX)

    return ChatPromptTemplate.from_messages([
        system_msg,
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ── Agent principal ───────────────────────────────────────────────────────────

async def run_sql_agent(question: str) -> Dict[str, Any]:
    """
    Exécute le SQL Agent sur une question en langage naturel.
    Retourne un dict compatible avec la structure de retour de ask_question().
    """
    logger.info("=" * 60)
    logger.info("[SQL Agent] ▶ DÉMARRAGE")
    logger.info("[SQL Agent] Question : %s", question)

    # ── Étape 1 : connexion Oracle ────────────────────────────────────────────
    logger.info("[SQL Agent] Étape 1/5 — Connexion Oracle")
    db = OracleClient.get_db()
    if db is None:
        logger.error("[SQL Agent] ✗ Oracle DB indisponible — agent non exécuté")
        return {**_FALLBACK_RESULT, "answer": _MSG_UNAVAILABLE}
    logger.info("[SQL Agent] ✓ Oracle connecté — tables autorisées : %s", OracleClient.ALLOWED_TABLES)

    try:
        # ── Étape 2 : Few-Shot selector ───────────────────────────────────────
        logger.info("[SQL Agent] Étape 2/5 — Sélection des exemples Few-Shot")
        selector = _get_few_shot_selector()
        if selector is not None:
            try:
                selected = selector.select_examples({"input": question})
                logger.info("[SQL Agent] ✓ %d exemple(s) sélectionné(s) :", len(selected))
                for i, ex in enumerate(selected, 1):
                    logger.info("[SQL Agent]   [%d] Q: %s", i, ex.get("input", "")[:100])
                    logger.info("[SQL Agent]   [%d] SQL: %s", i, ex.get("query", "")[:150])
            except Exception as e:
                logger.warning("[SQL Agent] ✗ Échec sélection few-shot : %s", e)
        else:
            logger.warning("[SQL Agent] ✗ FewShotSelector non disponible — prompt sans exemples")

        # ── Étape 3 : construction du prompt ─────────────────────────────────
        logger.info("[SQL Agent] Étape 3/5 — Construction du ChatPromptTemplate")
        llm    = LLMClient().model
        prompt = _build_agent_prompt(selector)
        logger.info("[SQL Agent] ✓ Prompt construit — messages : %d", len(prompt.messages))

        # ── Étape 4 : création de l'agent ─────────────────────────────────────
        logger.info("[SQL Agent] Étape 4/5 — Création du create_sql_agent (openai-tools)")
        agent = create_sql_agent(
            llm=llm,
            db=db,
            prompt=prompt,
            agent_type="openai-tools",
            verbose=True,
        )
        logger.info("[SQL Agent] ✓ Agent créé — lancement de ainvoke")

        # ── Étape 5 : exécution ReAct ─────────────────────────────────────────
        logger.info("[SQL Agent] Étape 5/5 — Boucle ReAct en cours...")
        result = await agent.ainvoke({"input": question})

        output: str = result.get("output", "")
        logger.info("[SQL Agent] ✓ Réponse finale (%d chars) : %s", len(output), output[:300])

        if "HORS_PERIMETRE" in output:
            logger.warning("[SQL Agent] ✗ Réponse hors périmètre détectée")
            logger.info("=" * 60)
            return {**_FALLBACK_RESULT, "answer": _MSG_OUT_OF_SCOPE}

        logger.info("[SQL Agent] ✓ FIN — réponse transmise au client")
        logger.info("=" * 60)
        return {**_FALLBACK_RESULT, "answer": output}

    except Exception as e:
        logger.error("[SQL Agent] ✗ ERREUR non gérée : %s", e, exc_info=True)
        logger.info("=" * 60)
        return {**_FALLBACK_RESULT, "answer": _MSG_TIMEOUT}
