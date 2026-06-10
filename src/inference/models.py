from pydantic import BaseModel


class QueryMetaData(BaseModel):
    source_url: str
    title: str = "Titre inconnu"
    date_posted: str = "Date inconnue"
