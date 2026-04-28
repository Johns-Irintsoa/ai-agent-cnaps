
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import config_env # Initialisation du cache /tmp

import os
import requests
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from .AIModel import DocumentValidator
from LLM.llm import LLMClient

# 3. PIPELINE DE TRAITEMENT
import os
from docling.document_converter import DocumentConverter

def process_unstructured_data(directory_path: str):
    """
    Orchestre les 3 méthodes : 
    1. Docling (Multimodal) 
    2. Heuristiques 
    3. Intent & Quality (LLM)
    """
    # Initialisation
    llm_client = LLMClient()
    validator = DocumentValidator(llm_client)
    converter = DocumentConverter()
    
    results = {
        "accepted": [],
        "rejected": []
    }

    if not os.path.exists(directory_path):
        raise ValueError(f"Le répertoire {directory_path} n'existe pas.")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue

        try:

            if file_path.lower().endswith('.doc') or file_path.lower().endswith('.xls'):
                raise ValueError(f"{filename}:: extension non supportée (.doc ou .xls).")

            # ÉTAPE 1 : Ingestion Multimodale (Docling)
            conversion_result = converter.convert(file_path)
            markdown_content = conversion_result.document.export_to_markdown()

            # ÉTAPE 2 : Filtrage Heuristique
            if len(markdown_content) < 300:
                results["rejected"].append({"file": filename, "reason": "Contenu trop court"})
                continue

            # ÉTAPE 3 : Intent & Quality (Validation LLM)
            decision = validator.validate(markdown_content, filename)

            doc_info = {
                "file": filename,
                "category": decision.category,
                "reason": decision.reason
            }

            if decision.is_useful:
                doc_info["content"] = markdown_content
                results["accepted"].append(doc_info)
            else:
                results["rejected"].append(doc_info)

        except Exception as e:
            results["rejected"].append({"file": filename, "reason": f"Erreur : {str(e)}"})

    return results