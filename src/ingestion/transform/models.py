from typing import Any, Dict

from pydantic import BaseModel


class WebPageContentChunked(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]
