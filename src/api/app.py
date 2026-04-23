from fastapi import FastAPI, HTTPException
from llm import LLMClient
from api.schemas import ChatRequest, ChatResponse, IndexRequest, IndexResponse, AskRequest, AskResponse, WebIngestionResponse, WebLoadResponse

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
