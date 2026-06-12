from typing import List, Optional

from pydantic import BaseModel

class WebPageFromJSON(BaseModel):
    url: str
    classes: Optional[List[str]] = None
    item_classes: Optional[List[str]] = None
    pagination_selector: Optional[str] = None
    is_contained_list: Optional[bool] = False

class WebPageContent(BaseModel):
    """" Represente une page web à scraper, avec son URL et les classes CSS à cibler pour l'extraction. """
    url: str
    classes: Optional[List[str]] = None

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
