import weaviate
from weaviate.classes.generate import GenerativeConfig

# step 1: Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")

# Step 2.1: Connect to your local Weaviate instance
with weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=8080,        # Port interne à Docker
    grpc_host=WEAVIATE_HOST,
    grpc_port=50051,        # Port gRPC interne
    http_secure=False,          # Pas de TLS pour une connexion locale
    grpc_secure=False            # Pas de TLS pour gRPC
) as client:
    
    # Step 2.2: Use this collection
    movies = client.collections.use("Movie")

    # Step 2.3: Perform RAG with on NearText results
    response = movies.generate.near_text(
        query="sci-fi",
        limit=1,
        grouped_task="Write a tweet with emojis about this movie.",
        # CORRECTION ICI : generative_config au lieu de generate_config
        generative_provider=GenerativeConfig.ollama(
            api_endpoint="http://ollama:11434",
            model="mistral:7b-instruct-q4_K_M",
        ),
    )

    print(response.generative.text)  # Inspect the results