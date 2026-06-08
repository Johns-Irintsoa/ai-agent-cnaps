from typing import List

from pydantic import BaseModel


class WebPageContent(BaseModel):
    """" Represente une page web à scraper, avec son URL et les classes CSS à cibler pour l'extraction. """
    url: str
    classes: List[str] = []
    pagination_selector: str | None = None
    is_paginated: bool = False

class WebPageContentConverted(BaseModel):
    """
    Représente la page web retourner version md avec source
    """

    url: str

class WebPageContentMd(BaseModel):
    """
    Représente la page web retourner version md avec source
    """

    url: str
    content: str


class WebPageMetadata(BaseModel):
    source_url: str
    title: str = "Titre inconnu"
    date_posted: str = "Date inconnue"


class WebPageContentExtracted(BaseModel):
    contenu_md: str
    metadata: WebPageMetadata
