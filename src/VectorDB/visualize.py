import chromadb
import os
from ..db.chroma_client import ChromaClient

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

    print(f"📦 Total chunks : {collection.count()}\n")

    # ── get() par metadata ───────────────────────────────────────────────────
    sig = collection.get(
        where={"is_signature_block": True},
        include=["documents", "metadatas"],
    )
    print(f"🔏 Chunks signature : {len(sig['ids'])}")
    for doc, meta in zip(sig["documents"], sig["metadatas"]):
        print(f"  → {meta.get('source')} | {doc[:200]}")

    # ── get() full-text ──────────────────────────────────────────────────────
    for keyword in ["ampliation", "22 MAY", "Antananarivo"]:
        r = collection.get(
            where_document={"$contains": keyword},
            include=["documents"],
        )
        status = "✅" if r["ids"] else "❌"
        print(f"{status} '{keyword}' → {len(r['ids'])} chunk(s)")
        for doc in r["documents"]:
            print(f"   {doc[:150]}")


if __name__ == "__main__":
    debug_chroma_simple()
