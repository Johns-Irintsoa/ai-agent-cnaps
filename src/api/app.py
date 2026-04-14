from fastapi import FastAPI, HTTPException
from llm import OllamaLLM
from api.schemas import ChatRequest, ChatResponse, IndexRequest, IndexResponse, AskRequest, AskResponse
from vector_database.scrap.scrapper import (
    load_page,
    split_text,
    index_docs,
    answer_question,
)

app = FastAPI(title="AI Agent CNAPS")

_llm = OllamaLLM()


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
