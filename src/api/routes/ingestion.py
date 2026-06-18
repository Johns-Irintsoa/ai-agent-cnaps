import asyncio
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..schemas import (
    PDFLoadRequest, WebIngestionResponse, WebLoadResponse,
    IngestURLRequest,
)
from ...ingestion.transform.service import transform_pdf, transform_web_page, transform_url

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/ingestion/web-pages", response_model=WebIngestionResponse)
async def ingest_web_pages() -> WebIngestionResponse:
    try:
        result = transform_web_page()
        if result is None:
            raise HTTPException(status_code=500, detail="Ingestion des pages web echouee.")
        from ...inference.bm25_retriever import bm25_index
        asyncio.get_event_loop().run_in_executor(None, bm25_index.refresh)
        return WebIngestionResponse(status="success", message="Pages web CNaPS ingrees et indexees dans ChromaDB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    # if not file.filename.endswith(".pdf"):
    #     raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chunk_count = transform_pdf(str(file_path))
        if chunk_count is None:
            raise HTTPException(status_code=500, detail="Le traitement du PDF a échoué.")
        from ...inference.bm25_retriever import bm25_index
        asyncio.get_event_loop().run_in_executor(None, bm25_index.refresh)
        return JSONResponse(content={"message": f"{chunk_count} chunk(s) indexé(s) depuis '{file.filename}'", "status": "success"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
    finally:
        if file_path.exists():
            os.remove(file_path)

@router.post("/load/pdfs", response_model=WebLoadResponse)
async def load_pdfs_data(request: PDFLoadRequest) -> WebLoadResponse:
    from ...ingestion.load.Service import load_pdf_data
    try:
        docs = load_pdf_data(request.pdf_path)
        return WebLoadResponse(
            status="success",
            message=f"{len(docs)} document(s) charges depuis {request.pdf_path}",
            documents_loaded=len(docs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/url", response_model=WebIngestionResponse)
async def ingest_url(request: IngestURLRequest) -> WebIngestionResponse:
    """Parse une URL avec les classes CSS fournies et ingère le contenu dans ChromaDB."""
    try:
        collection_name = os.getenv("COLLECTION_NAME", "rag_cnaps")
        count = await asyncio.to_thread(transform_url, request.url, request.classes, collection_name)
        if count > 0:
            from ...inference.bm25_retriever import bm25_index
            await asyncio.to_thread(bm25_index.refresh)
        return WebIngestionResponse(
            status="success",
            message=f"{count} chunk(s) ingéré(s) depuis {request.url}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
