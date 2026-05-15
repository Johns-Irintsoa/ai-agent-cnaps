import asyncio

from src.inference.query_retriever import get_query_vector, search_similar_documents
from src.inference.reranking import get_reranked_documents
from src.inference.prompting import generate_answer, generate_answer_multi_query
from src.inference.evaluation import print_evaluation, evaluate_answer

from src.inference.multi_query_retriever import  multi_query_retriever_async
from typing import Dict, Any
from .cache.semantic_cache import semantic_cache


async def ask_question(user_query: str)-> Dict[str, Any]:
    """
    Fonction principale pour traiter une question utilisateur : 
    0. Verifier le cache sémantique pour une question similaire (optionnel, à intégrer)
    1. Récupérer les documents pertinents via multi-query retrieval
    2. Générer une réponse à partir de ces documents
    3. Évaluer la qualité de la réponse
    """
    # Vérifier le cache
    cached = await semantic_cache.get(user_query)
    if cached:
        answer, _ = cached
        return {"answer": answer, "evaluation": {}, "from_cache": True}

    docs = await multi_query_retriever_async(
        query=user_query,
        top_k_per_query=4,
        top_k_final=4,
    )
    # Generer une réponse à partir des documents multi-query
    answer = await asyncio.to_thread(generate_answer_multi_query, user_query, docs)
    # Mettre en cache en arrière‑plan
    asyncio.create_task(semantic_cache.set(user_query, answer, {"docs_count": len(docs)}))
    return {"answer": answer, "evaluation": {}, "from_cache": False}

# if __name__ == "__main__":

#     # 1- TRADITIONAL USER QUERY (un seul passage de retrieval) — pour comparaison avec le multi-query
    
#     # user_query = "Quelle est l'article 5 du decret n 2015-809?"
#     # user_query = "qui etait le secretaire general adjoint du gouvernement qui a signe le decret N2015-809?"
#     # user_query = "Qui sont les autorités qui ont été present et accorde ce decret, mentionne leur noms?"
#     # user_query = "Quand ce decret a ete ?"
#     # user_query = "Est ce que tu saurais m'expliquer le tableau dans le secteur agricole dans le decret N°2015-809?"

#     # Question synonyme sur la date de l'ampliation conforme du decret
#     # user_query = "Quelle est la date de l'ampliation figurant sur ce document?";
#     # user_query = "À quel moment ce document a-t-il été officiellement dupliqué le decret n 2015-809?"
#     # user_query = "Quelle est la date de délivrance du duplicata officiel du decret n 2015-809 ?"
#     # user_query = "À quelle date le duplicata du decret N°2015-809 a-t-il été établi à Antananarivo ?"
#     # user_query = "Pourriez-vous me préciser la date de signature du duplicata du decret N°2015-809 ?"


    
#     # # 1. Obtenir uniquement le vecteur (si besoin pour debug ou autre service)
#     # vector = get_query_vector(user_query)
    
#     # # 2. Obtenir directement les documents pertinents
#     # documents = search_similar_documents(user_query)

#     # document_reranked = get_reranked_documents(user_query, documents, k_final=3)
    
#     # print("\n\n\n\n\n--- Initial Documents ---")
#     # for doc in documents:
#     #     print(f"\nSource: {doc.metadata.get('source')}")
#     #     print(f"Contenu: {doc.page_content}")
    
#     # print("\n\n\n\n\n--- Reranked Documents ---")
#     # for doc in document_reranked:
#     #     print(f"\nSource: {doc.metadata.get('source')}")
#     #     print(f"Contenu: {doc.page_content}")

#     # print("\n--- Réponse du LLM ---")
#     # answer = generate_answer(user_query, document_reranked)
#     # print(answer)

#     # print("\n--- Évaluation de la réponse ---")
#     # scores = evaluate_answer(user_query, document_reranked, answer)
#     # print_evaluation(scores)

#     # 2- MULTI-QUERY RETRIEVAL (génère plusieurs reformulations de la question pour un retrieval plus robuste)
#     user_query = "Quelle est la date de l'ampliation conforme figurant sur le decret N°2015-809?"
    
#     docs = multi_query_retriever(
#         query=user_query,
#         top_k_per_query=4,
#         top_k_final=4,
#     )

#         # print(f"\n{'='*60}")
    
#     print(f"✅ {len(docs)} chunks retournés\n")
#     for i, doc in enumerate(docs, 1):
#         print(f"[{i}] {doc.metadata.get('source', '?')} "
#                 f"| table={doc.metadata.get('is_table', False)}")
#         print(f"    {doc.page_content.strip()}...\n")

#     # # 2.1 generer une réponse à partir des documents multi-query
#     # answer = generate_answer_multi_query(user_query, docs)
#     # print("\n--- Réponse du LLM ---")
#     # print(answer)

#     # # 2.2 évaluer la réponse
#     # scores = evaluate_answer(user_query, docs, answer)
#     # print("\n--- Évaluation de la réponse ---")
#     # print_evaluation(scores)


