import uuid
import json
from datetime import datetime
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from parsing import _pdf_docling
from pathlib import Path

def chuncking_md_data(md_text, filename="document_source", max_chunk_size=1000, chunk_overlap=100):
    """
    Découpe le markdown en chunks structurés (Hybride : Titres + Taille).
    
    Args:
        md_text (str): Le contenu markdown issu de docling.
        filename (str): Nom du fichier source pour les métadonnées.
        max_chunk_size (int): Taille maximum de caractères par chunk.
        chunk_overlap (int): Chevauchement entre les chunks pour garder le contexte.
    """
    
    # 1. Premier découpage : Basé sur la structure des titres (#, ##, ###)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    header_splits = markdown_splitter.split_text(md_text)

    # 2. Deuxième découpage : Sécurité pour les sections trop longues
    # On utilise RecursiveCharacter pour ne pas couper les phrases/mots brutalement
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_json_list = []
    
    # 3. Traitement et enrichissement
    for split in header_splits:
        # On recoupe chaque section si elle dépasse max_chunk_size
        sub_chunks = text_splitter.split_documents([split])
        
        for sub_chunk in sub_chunks:
            chunk_id = str(uuid.uuid4())
            
            # Reconstruction de la hiérarchie (ex: Titre > Sous-titre)
            hierarchy = " > ".join([val for key, val in sub_chunk.metadata.items() if "Header" in key])
            
            # Création de l'item selon votre structure
            current_index = len(final_json_list)
            chunk_item = {
                "chunk_id": chunk_id,
                "content": sub_chunk.page_content,
                "metadata": {
                    "source": filename,
                    "page_numbers": [], # Note: Nécessite l'objet JSON de Docling pour être exact
                    "creation_date": datetime.now().isoformat(),
                    "hierarchical_context": hierarchy or "Root",
                    "chunk_index": current_index + 1,
                    "previous_chunk_id": final_json_list[current_index - 1]["chunk_id"] if current_index > 0 else None,
                    "next_chunk_id": None,
                    "contains_table": "|" in sub_chunk.page_content
                }
            }

            # Mise à jour du lien "next" du chunk précédent
            if current_index > 0:
                final_json_list[current_index - 1]["metadata"]["next_chunk_id"] = chunk_id
                
            final_json_list.append(chunk_item)

    # Mise à jour du total_chunks pour tous les éléments
    total_count = len(final_json_list)
    for item in final_json_list:
        item["metadata"]["total_chunks"] = total_count

    return final_json_list


# # Exemple d'usage :
# _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# # print(_PROJECT_ROOT)
# sample_file = _PROJECT_ROOT / "data/unstructured/pdf/265df2015-CNaPS_600fd1b4ca3383.50730780.pdf"
# sample_md_text = _pdf_docling(sample_file)

# text_result = chuncking_md_data(sample_md_text, filename="265df2015-CNaPS_600fd1b4ca3383.50730780.pdf")
# print(text_result)