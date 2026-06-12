from pydantic import BaseModel
from typing import List


class SourceItem(BaseModel):
    source_url: str
    title: str = "Titre inconnu"
    date_posted: str = "Date inconnue"


class QueryMetaData(BaseModel):
    title: str = "Titre inconnu"
    date_posted: str = "Date inconnue"
    sources: List[SourceItem] = []
