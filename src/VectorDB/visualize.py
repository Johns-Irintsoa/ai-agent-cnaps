import chromadb
import os
import sys
from ..db.chroma_client import ChromaClient


def verify(
    query: str,
    collection_name: str = "rag_cnaps",
    k: int = 5,
):
    """
    Recherche sémantique dans ChromaDB à partir d'un texte libre.

    Génère l'embedding du query via le Model Runner puis interroge
    la collection avec similarity_search_with_relevance_scores.
    Affiche les k meilleurs chunks avec leur score, source et extrait.

    Usage (docker exec) :
        docker exec -it ai-agent-cnaps python -m src.VectorDB.visualize verify "votre question"
    """
    from .initialize import get_vector_db
    from ..models.embedding import EmbeddingManager

    print(f"\nRequete : {query!r}")
    print(f"Collection : {collection_name} | top-k : {k}\n")

    embedding_model = EmbeddingManager().model
    db = get_vector_db(embedding_model)

    results = db.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        print("Aucun document trouve.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        source  = doc.metadata.get("source", "—")
        page    = doc.metadata.get("page", "—")
        excerpt = doc.page_content[:300].replace("\n", " ")
        print(f"[{rank}] score={score:.4f}  source={source}  page={page}")
        print(f"     {excerpt}")
        print()


def debug_chroma_simple(
    collection_name: str = "rag_cnaps",
):
    """
    Recherche les chunks liés à l'ampliation sans LLM.
    Utilise uniquement get() avec filtres metadata et full-text.
    Aucun embedding, aucun LLM requis.
    Se connecte via HttpClient (Docker) ou PersistentClient (local) selon CHROMA_HOST.
    """
    client = ChromaClient.get_client()
    collection = client.get_collection(name=collection_name)

    print(f"Total chunks : {collection.count()}\n")

    # ── get() par metadata ───────────────────────────────────────────────────
    sig = collection.get(
        where={"is_signature_block": True},
        include=["documents", "metadatas"],
    )
    print(f"Chunks signature : {len(sig['ids'])}")
    for doc, meta in zip(sig["documents"], sig["metadatas"]):
        print(f"  → {meta.get('source')} | {doc[:200]}")

    # ── get() full-text ──────────────────────────────────────────────────────
    for keyword in ["ampliation", "22 MAY", "Antananarivo"]:
        r = collection.get(
            where_document={"$contains": keyword},
            include=["documents"],
        )
        status = "OK" if r["ids"] else "KO"
        print(f"{status} '{keyword}' → {len(r['ids'])} chunk(s)")
        for doc in r["documents"]:
            print(f"   {doc[:150]}")


if __name__ == "__main__":
    # Usage :
    #   python -m src.VectorDB.visualize                        → debug_chroma_simple
    #   python -m src.VectorDB.visualize verify "ma question"   → verify
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        if len(sys.argv) < 3:
            print("Usage : python -m src.VectorDB.visualize verify \"votre question\"")
            sys.exit(1)
        query = sys.argv[2]
        k     = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        verify(query, k=k)
    else:
        debug_chroma_simple()
