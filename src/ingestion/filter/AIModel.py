from pydantic import BaseModel, Field
from LLM.llm import LLMClient 

# Définition du schéma de sortie
class RAGDecision(BaseModel):
    is_useful: bool = Field(description="Le document est-il pertinent pour le RAG ?")
    category: str = Field(description="Type de doc (Technique, Rapport, Facture, Bruit)")
    reason: str = Field(description="Justification rapide")

class DocumentValidator:
    def __init__(self, client: LLMClient):
        # On lie le schéma Pydantic au modèle via la propriété .model
        self.structured_llm = client.model.with_structured_output(RAGDecision)

    def validate(self, content_sample: str, filename: str) -> RAGDecision:
        prompt = f"""Analyse ce contenu extrait du fichier '{filename}'. 
        Détermine s'il est utile pour une base de connaissances RAG.
        
        CONTENU :
        {content_sample[:2000]}"""
        
        return self.structured_llm.invoke(prompt)