from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

from ..models.llm import LLMClient

_llm_client = LLMClient()

PROMPT_TEMPLATE =  """Tu es Lucy, assistante CNaPS Madagascar. Réponds uniquement aux questions sur la CNaPS (retraites, cotisations, prestations, affiliation). Hors sujet : refuse poliment.

CONTEXTE (DOCUMENTS CNaPS) :
{context}

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir du contexte ci-dessus. Si l'information exacte n'est pas dans le contexte, dis : "Je n'ai pas l'information nécessaire. Veuillez contacter un agent CNaPs."
2. Ne suppose rien, n'invente pas, ne généralise pas.
3. Sois précise, utilise des listes si utile.
4. Réponds en français.
5. **Limite ta réponse à 3 phrases maximum, sauf si la question exige une liste d'étapes (max 5 puces).** Évite les détails superflus.

QUESTION : {question}
RÉPONSE (courte et utile) :"""

# 1 Generate answer for simple retrieval
def generate_answer(query: str, reranked_docs: List[Document]) -> str:
    """
    Prend la query et les documents rerankés, construit le prompt et invoque le LLM.
    """
    # List[Document] → string
    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Injection dans le template
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    
    # Invocation du LLM
    return _llm_client.invoke(prompt)

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