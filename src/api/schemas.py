from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class IndexRequest(BaseModel):
    url: str


class IndexResponse(BaseModel):
    indexed_chunks: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class WebIngestionResponse(BaseModel):
    status: str
    message: str
