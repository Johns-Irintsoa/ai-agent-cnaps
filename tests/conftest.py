import json
import pytest
from pathlib import Path

from ingestion.scraping.models import WebPageContentExtracted, WebPageMetadata


CNAPS_HISTORIQUE_URL = "https://www.cnaps.mg/fr/cnaps/historique"
CNAPS_FAQ_URL = "https://www.cnaps.mg/fr/faq"
CNAPS_MISSIONS_URL = "https://www.cnaps.mg/fr/cnaps/missions-et-engagements"


@pytest.fixture
def sample_page() -> WebPageContentExtracted:
    return WebPageContentExtracted(
        contenu_md="# Historique CNaPS\n\nParagraphe un.\n\nParagraphe deux.",
        metadata=WebPageMetadata(
            source_url=CNAPS_HISTORIQUE_URL,
            title="Historique CNaPS",
            date_posted="2024-01-01",
        ),
    )


@pytest.fixture
def minimal_json_file(tmp_path: Path) -> Path:
    data = {
        "cnaps_urls": [
            {"url": "https://example.com/a", "classes": ["cls-a"], "is_paginated": False},
            {"url": "https://example.com/b", "classes": [], "is_paginated": False},
        ]
    }
    path = tmp_path / "urls.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
