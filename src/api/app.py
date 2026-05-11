# from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, UploadFile, File, HTTPException

from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import JSONResponse
import shutil
import os

from src.inference import multi_query_retriever
# from models.llm import LLMClient
from .schemas import RAGResponse, ChatRequest, ChatResponse, IndexRequest, IndexResponse, AskRequest, AskResponse, WebIngestionResponse, WebLoadResponse, FileLoadTestResponse, DocumentContent, IngestionRequest, PDFLoadRequest
from ..ingestion.filter.functions import process_unstructured_data
from src.ingestion.transform.service import transform_pdf
from src.inference.prompting import generate_answer
from src.inference.evaluation import evaluate_answer
from src.inference.service import ask_question


app = FastAPI(title="AI Agent CNAPS")

# Dossier temporaire pour stocker les PDF reçus
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/scraper/index", response_model=IndexResponse)
def scraper_index(request: IndexRequest) -> IndexResponse:
    try:
        documents = load_page(request.url)
        chunks = split_text(documents)
        index_docs(chunks)
        return IndexResponse(indexed_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scraper/ask", response_model=AskResponse)
def scraper_ask(request: AskRequest) -> AskResponse:
    try:
        answer = answer_question(request.question)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingestion/scrap", response_model=WebIngestionResponse)
def scrap_web() -> WebIngestionResponse:
    from ingestion.scrap.Service import from_web
    result = from_web()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return WebIngestionResponse(status=result["status"], message=result["message"])

@app.post("/ingestion", response_model=WebIngestionResponse)
def ingestion_pipeline() -> WebIngestionResponse:
    from ingestion.scrap.Service import from_web
    result = from_web()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return WebIngestionResponse(status=result["status"], message=result["message"])


@app.post("/ingestion/scrap/v1", response_model=WebIngestionResponse)
def scrap_web_data() -> WebIngestionResponse:
    from ingestion.scrap.Service import scrap_from_web
    result = scrap_from_web()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return WebIngestionResponse(status=result["status"], message=result["message"])


@app.post("/ingestion/load/web-data", response_model=WebLoadResponse)
def ingestion_load_web_data() -> WebLoadResponse:
    from src.ingestion.load.Service import load_web_data
    try:
        docs = load_web_data()
        return WebLoadResponse(
            status="success",
            message=f"{len(docs)} document(s) charges depuis cnaps_urls.json",
            documents_loaded=len(docs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingestion/load/test", response_model=FileLoadTestResponse)
def ingestion_load_test() -> FileLoadTestResponse:
    from ingestion.load.Service import load_pdf_with_image
    try:
        docs = load_pdf_with_image()
        return FileLoadTestResponse(
            status="success",
            documents_loaded=len(docs),
            documents=[
                DocumentContent(page_content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ingestion/filter")
async def ingest_documents(request: IngestionRequest):
    """
    Endpoint pour lancer le filtrage et l'ingestion des documents.
    """
    try:
        # On appelle l'orchestrateur
        output = process_unstructured_data(request.directory_path)
        
        return {
            "status": "success",
            "summary": {
                "total_accepted": len(output["accepted"]),
                "total_rejected": len(output["rejected"])
            },
            "data": output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/load/pdfs", response_model=WebLoadResponse)
async def load_pdfs_data(request: PDFLoadRequest) -> WebLoadResponse:
    from ingestion.load.Service import load_pdf_data
    try:
        docs = load_pdf_data(request.pdf_path)
        return WebLoadResponse(
            status="success",
            message=f"{len(docs)} document(s) charges depuis {request.pdf_path}",
            documents_loaded=len(docs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour l'ingestion de PDF via upload CHROMA db

@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Endpoint pour uploader un PDF et lancer le pipeline RAG.
    """
    # 1. Validation du type de fichier
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    file_path = UPLOAD_DIR / file.filename

    try:
        # 2. Sauvegarde temporaire du fichier sur le disque
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Appel de transform_pdf
        result = transform_pdf(str(file_path))

        if result is None:
            raise HTTPException(status_code=500, detail="Le traitement du PDF a échoué.")

        return JSONResponse(content={
            "message": f"Document '{file.filename}' indexé avec succès",
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

    finally:
        # 4. Nettoyage : supprimer le fichier temporaire après traitement
        if file_path.exists():
            os.remove(file_path)

@app.post("/ask", response_model=RAGResponse)
async def ask(request: ChatRequest):
    """
    Endpoint qui effectue un retrieval multi-query,
    génère une réponse et l'évalue.
    """
    try:
        user_query = request.message
        # Appel de la fonction modifiée
        result = ask_question(user_query)
        return RAGResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la requête: {str(e)}")