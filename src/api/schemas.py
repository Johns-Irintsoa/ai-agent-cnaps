from pydantic import BaseModel
from typing import Dict, Any, List, Literal, Optional
from ..inference.models import QueryMetaData

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str

class RAGResponse(BaseModel):
    answer: str
    metadata: Optional[QueryMetaData] = None


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


class WebLoadResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int


class DocumentContent(BaseModel):
    page_content: str
    metadata: dict


class FileLoadTestResponse(BaseModel):
    status: str
    documents_loaded: int
    documents: list[DocumentContent]


class IngestionRequest(BaseModel):
    directory_path: str

class PDFLoadRequest(BaseModel):
    pdf_path: str


class DeleteBySourceURLRequest(BaseModel):
    source_url: str


class DeleteByIdsRequest(BaseModel):
    ids: List[str]


class DeleteResponse(BaseModel):
    status: str
    deleted_count: int
    message: str


class IngestURLRequest(BaseModel):
    url: str
    classes: List[str] = []

