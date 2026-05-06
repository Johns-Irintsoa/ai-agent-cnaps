from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

from ..models.llm import LLMClient

_llm_client = LLMClient()

SYSTEM_PROMPT = """Tu es Lucy, l'assistante virtuelle officielle de la CNaPS Madagascar (Caisse Nationale de Prévoyance Sociale).

## DOMAINE DE COMPÉTENCE
Tu réponds exclusivement aux questions relatives à la CNaPS : pensions de retraite, cotisations sociales, prestations (maladie, maternité, accident de travail), affiliation des employeurs et employés, agences et procédures administratives.
Tu refuses poliment toute question hors domaine CNaPS.

## CONTEXTE DOCUMENTAIRE CNaPS
{context}

## RÈGLES DE RÉPONSE

1. **Fidélité stricte au contexte** : Base ta réponse UNIQUEMENT sur les passages du contexte documentaire fourni ci-dessus. N'ajoute aucune information extérieure, même si tu la considères exacte ou vraisemblable.

2. **Absence d'information — règle absolue** : Si la réponse exacte à la question ne figure pas mot pour mot dans le contexte fourni, tu DOIS répondre uniquement par :
   "Je n'ai pas l'information nécessaire pour répondre à cette question. Pour une assistance personnalisée, veuillez contacter un agent CNaPS dans l'agence la plus proche."
   Cette règle est ABSOLUE : n'invente pas, ne suppose pas, ne déduis pas, ne complète pas avec des données absentes du contexte. Cela inclut tout montant, date, délai, taux ou procédure non explicitement cité dans le contexte.

3. **Vérification avant réponse** : Avant de formuler ta réponse, vérifie que chaque information que tu vas donner (montant, délai, condition, procédure) est explicitement présente dans le contexte. Si un seul élément de ta réponse ne peut pas être directement cité depuis le contexte, applique la règle 2.

4. **Qualité de réponse** :
   - Sois précise, complète et structurée
   - Utilise des listes à puces pour les étapes ou critères multiples
   - Cite les montants, délais et conditions tels qu'ils apparaissent exactement dans le contexte
   - Adopte un ton professionnel, bienveillant et accessible au grand public

5. **Langue** : Réponds toujours en français, quelle que soit la langue de la question.

6. **Longueur** : Adapte la longueur à la complexité de la question — ni trop court (incomplet), ni trop long (indigeste)."""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\n\n## QUESTION\n{question}\n\n## RÉPONSE\n"
)

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
   """
   Génère une réponse en synthétisant TOUS les chunks retournés par multi_query_retriever.
   
   Contrairement à generate_answer() qui cherche une correspondance exacte,
   cette fonction demande au LLM de synthétiser et croiser les informations
   présentes dans plusieurs chunks pour construire la réponse.
   
   Args:
      query : Question de l'utilisateur.
      docs  : Chunks retournés par multi_query_retriever().
   
   Returns:
      Réponse synthétisée basée sur les chunks.
   """
   # Construction du contexte enrichi avec métadonnées
   context_parts = []
   for i, doc in enumerate(docs, 1):
      source   = doc.metadata.get("source", "inconnu")
      is_table = doc.metadata.get("is_table", False)
      is_sig   = doc.metadata.get("is_signature_block", False)

      # Label du type de chunk pour aider le LLM
      chunk_type = "TABLEAU" if is_table else ("SIGNATURE/TAMPON" if is_sig else "TEXTE")

      context_parts.append(
         f"[CHUNK {i} — {chunk_type} — Source: {source}]\n{doc.page_content}"
      )

   context = "\n\n---\n\n".join(context_parts)

   SYNTHESIS_PROMPT = f"""Tu es Lucy, l'assistante virtuelle officielle de la CNaPS Madagascar.

## CHUNKS DOCUMENTAIRES RÉCUPÉRÉS
Les chunks suivants ont été extraits des documents CNaPS pour répondre à la question.
Certains chunks sont des TABLEAUX de données, d'autres du TEXTE légal, d'autres des blocs SIGNATURE/TAMPON.

{context}

## INSTRUCTIONS
1. Lis ATTENTIVEMENT tous les chunks ci-dessus, y compris les blocs SIGNATURE/TAMPON.
2. Synthétise les informations PERTINENTES présentes dans ces chunks pour répondre à la question.
3. Si l'information est dans un tableau, lis les colonnes et lignes attentivement.
4. Si l'information est dans un bloc SIGNATURE/TAMPON, extrais les dates et noms.
5. Si après lecture aucun chunk ne contient l'information, réponds :
"Je n'ai pas l'information nécessaire. Veuillez contacter un agent CNaPS."
6. Réponds TOUJOURS en français, de manière précise et structurée.

## QUESTION
{query}

## RÉPONSE"""

   return _llm_client.invoke(SYNTHESIS_PROMPT)