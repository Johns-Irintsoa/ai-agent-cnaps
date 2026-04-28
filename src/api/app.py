from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from LLM.llm import LLMClient
from api.schemas import ChatRequest, ChatResponse, IndexRequest, IndexResponse, AskRequest, AskResponse, WebIngestionResponse, WebLoadResponse, FileLoadTestResponse, DocumentContent, IngestionRequest, PDFLoadRequest
from ingestion.filter.functions import process_unstructured_data

app = FastAPI(title="AI Agent CNAPS")

_llm = LLMClient()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = _llm.invoke(request.message)
    return ChatResponse(response=answer)


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
