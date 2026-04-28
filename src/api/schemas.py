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
