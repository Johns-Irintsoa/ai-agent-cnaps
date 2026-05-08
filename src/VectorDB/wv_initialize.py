import weaviate
from weaviate.classes.config import Configure
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration weaviate
WEAVIATE_HOST = os.getenv("WEAVIATE_URL") 
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")

# Utilisez connect_to_custom pour spécifier l'adresse interne du réseau Docker
with weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=8080,        # Port interne à Docker
    grpc_host=WEAVIATE_HOST,
    grpc_port=50051,        # Port gRPC interne
    http_secure=False,          # Pas de TLS pour une connexion locale
    grpc_secure=False            # Pas de TLS pour gRPC
) as client:
    
# Vérification de sécurité pour éviter de créer deux fois la collection
    if not client.collections.exists("Movie"):
        movies = client.collections.create(
            name="Movie",
            vector_config=Configure.Vectorizer.text2vec_transformers(model_name="all-MiniLM-L6-v2")
        )
    
    movies = client.collections.get("Movie")
    
    data_objects = [
        {"title": "The Matrix", "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.", "genre": "Science Fiction"},
        {"title": "Spirited Away", "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.", "genre": "Animation"},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.", "genre": "Fantasy"},
    ]

    movies = client.collections.use("Movie")
    with movies.batch.fixed_size(batch_size=200) as batch:
        for obj in data_objects:
            batch.add_object(properties=obj)

    print(f"Imported & vectorized {len(movies)} objects into the Movie collection")