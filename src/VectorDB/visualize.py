import chromadb

# ✅ Chemin correct trouvé
client = chromadb.PersistentClient(path="/app/vector_cnaps_db")

# Lister les collections
collections = client.list_collections()
print("Collections disponibles :")
for col in collections:
    print(f"  - {col.name}")

# Voir les données d'une collection
col = client.get_collection("rag_cnaps")
data = col.get(include=["documents", "metadatas"])

print(f"\nNombre de documents : {len(data['ids'])}")
for i, doc_id in enumerate(data['ids']):
    print(f"\n[{doc_id}]")
    print(f"  Document  : {data['documents'][i]}")
    print(f"  Métadonnée: {data['metadatas'][i]}")