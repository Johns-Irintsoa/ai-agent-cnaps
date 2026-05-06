import json
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ..models.llm import LLMClient

llm_client = LLMClient()

JUDGE_PROMPT = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="""Tu es un évaluateur expert et impartial de systèmes RAG.
Tu dois évaluer la qualité de la réponse générée par rapport au contexte documentaire fourni.

## CONTEXTE DOCUMENTAIRE
{context}

## QUESTION POSÉE
{question}

## RÉPONSE GÉNÉRÉE
{answer}

## INSTRUCTIONS D'ÉVALUATION

Évalue la réponse selon ces deux métriques :

1. **ACCURACY (Exactitude)** — Entre 0.0 et 1.0
   Mesure si les faits de la réponse sont corrects et présents dans le contexte.
   - 1.0 : Tous les faits sont exacts et vérifiables dans le contexte
   - 0.5 : Certains faits sont corrects, d'autres absents ou approximatifs
   - 0.0 : Les faits sont incorrects ou inventés (hallucination)

2. **F1_SCORE** — Entre 0.0 et 1.0
   Mesure l'équilibre entre la complétude et la précision de la réponse.
   - Precision : Les informations données sont-elles toutes pertinentes ?
   - Recall    : Les informations importantes du contexte sont-elles toutes couvertes ?
   - F1        : Moyenne harmonique des deux ( 2 * precision * recall / (precision + recall) )
   - 1.0 : Réponse complète et précise
   - 0.5 : Réponse partiellement complète ou partiellement précise
   - 0.0 : Réponse hors sujet ou vide

## FORMAT DE RÉPONSE OBLIGATOIRE
Réponds UNIQUEMENT avec ce JSON, sans texte avant ni après, sans backticks :
{{
    "accuracy": <float entre 0.0 et 1.0>,
    "f1_score": <float entre 0.0 et 1.0>,
    "accuracy_reason": "<explication courte>",
    "f1_reason": "<explication courte>"
}}"""
)


def evaluate_answer(
    query: str,
    reranked_docs: List[Document],
    generated_answer: str
) -> dict:
    """
    Évalue la qualité d'une réponse générée via LLM-as-a-Judge.
    Retourne un dictionnaire avec accuracy, f1_score et les justifications.
    """
    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])

    prompt = JUDGE_PROMPT.format(
        context=context,
        question=query,
        answer=generated_answer
    )

    raw_response = llm_client.invoke(prompt)

    try:
        # Nettoyage au cas où le LLM ajouterait des backticks malgré la consigne
        clean = raw_response.strip().replace("```json", "").replace("```", "")
        scores = json.loads(clean)
    except json.JSONDecodeError:
        print(f"⚠️ Réponse non parseable : {raw_response}")
        scores = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "accuracy_reason": "Erreur de parsing",
            "f1_reason": "Erreur de parsing"
        }

    return scores


def print_evaluation(scores: dict):
    """Affiche les scores de manière lisible."""
    print("\n--- Évaluation LLM-as-a-Judge ---")
    print(f"  Accuracy : {scores['accuracy']:.2f} — {scores['accuracy_reason']}")
    print(f"  F1 Score : {scores['f1_score']:.2f} — {scores['f1_reason']}")

    # Interprétation globale
    avg = (scores["accuracy"] + scores["f1_score"]) / 2
    if avg >= 0.8:
        verdict = "✅ Excellente réponse"
    elif avg >= 0.5:
        verdict = "⚠️ Réponse acceptable mais améliorable"
    else:
        verdict = "❌ Réponse insuffisante"
    print(f"  Verdict  : {verdict} (score moyen : {avg:.2f})")