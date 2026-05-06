import chromadb
import os

def debug_chroma_simple(
    collection_name: str = "rag_cnaps",
):
    """
    Recherche les chunks liés à l'ampliation sans LLM.
    Utilise uniquement get() avec filtres metadata et full-text.
    Aucun embedding, aucun LLM requis.
    """
    client = chromadb.PersistentClient(
        path=os.getenv("VECTOR_DB_DIR", "./vector_cnaps_db")
    )
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


# # ✅ Chemin correct trouvé
# client = chromadb.PersistentClient(path="/app/vector_cnaps_db")

# # Lister les collections
# collections = client.list_collections()
# print("Collections disponibles :")
# for col in collections:
#     print(f"  - {col.name}")

# # Voir les données d'une collection
# col = client.get_collection("rag_cnaps")
# data = col.get(include=["documents", "metadatas"])

# print(f"\nNombre de documents : {len(data['ids'])}")
# for i, doc_id in enumerate(data['ids']):
#     print(f"\n[{doc_id}]")
#     print(f"  Document  : {data['documents'][i]}")
#     print(f"  Métadonnée: {data['metadatas'][i]}")