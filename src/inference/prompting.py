from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

from ..models.llm import LLMClient

_llm_client = LLMClient()

PROMPT_TEMPLATE = """Tu es Lucy, assistante CNaPS Madagascar. Réponds uniquement aux questions sur la CNaPS (retraites, cotisations, prestations, affiliation). Hors sujet : refuse poliment.

CONTEXTE :
{context}

RÈGLES :
1. Réponds UNIQUEMENT depuis le contexte. Si absent : "Je n'ai pas l'information nécessaire. Veuillez contacter un agent CNaPS."
2. Ne suppose rien, n'invente rien.
3. Réponds en français.
4. Adapte la longueur au type de question :
   - Question simple (date, lieu, contact) → 1 à 2 phrases claires.
   - Question sur une procédure, condition ou calcul → explique avec des détails utiles, utilise une liste structurée (max 7 puces) ou 2 paragraphes.
   - Évite les répétitions et les détails hors sujet.
5. Si des sources sont disponibles, termine TOUJOURS ta réponse par :
   "Pour plus d'informations : {sources}"

QUESTION : {question}
RÉPONSE :"""

# 1 Generate answer for simple retrieval
def generate_answer(query: str, reranked_docs: List[Document]) -> tuple[str, dict]:
    """
    Prend la query et les documents rerankés, construit le prompt et invoque le LLM.
    Retourne (answer: str, tokens: dict) avec prompt_tokens, completion_tokens, total_tokens.
    """
    MAX_CHARS_PER_CHUNK = 1200
    context = "\n\n---\n\n".join([
        doc.page_content[:MAX_CHARS_PER_CHUNK] for doc in reranked_docs
    ])

    seen = set()
    sources = []
    for doc in reranked_docs:
        url = doc.metadata.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    sources_str = " | ".join(sources) if sources else ""

    prompt = PROMPT_TEMPLATE.format(context=context, question=query, sources=sources_str)
    return _llm_client.invoke_with_usage(prompt)

# 2 Generate response for multi-query retrieval (ex: question + 3 reformulations)
def generate_answer_multi_query(query: str, docs: List[Document]) -> str:
    # Construction du contexte (conserve les métadonnées mais sans les labels explicites)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    SYNTHESIS_PROMPT = f"""Lucy (assistante CNaPS). Réponds uniquement à partir des chunks ci-dessous.
Si l'information n'est pas dans les chunks, dis : "Je n'ai pas l'information. Contactez un agent CNaPS plus près de chez vous."

Chunks :
{context}

Question : {query}
Réponse courte (max 3 phrases ou 5 puces, en français) :"""

    return _llm_client.invoke(SYNTHESIS_PROMPT)