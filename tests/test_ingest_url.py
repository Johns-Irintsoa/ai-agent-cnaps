from unittest.mock import patch, MagicMock
from src.ingestion.scraping.models import WebPageContentExtracted, WebPageMetadata


@patch("src.ingestion.transform.service.embed_chunks")
@patch("src.ingestion.transform.service.chunking_md_html")
@patch("src.ingestion.transform.service._parse_html")
def test_transform_url_returns_chunk_count(mock_parse, mock_chunk, mock_embed):
    mock_parse.return_value = WebPageContentExtracted(
        contenu_md="# Titre\nContenu de la page",
        metadata=WebPageMetadata(source_url="https://cnaps.mg/page", title="Titre", date_posted="2024-01-01"),
    )
    mock_chunk.return_value = [MagicMock(), MagicMock()]
    mock_embed.return_value = MagicMock()

    from src.ingestion.transform.service import transform_url
    count = transform_url("https://cnaps.mg/page", ["content-article"], "rag_cnaps")

    mock_parse.assert_called_once()
    called_page = mock_parse.call_args[0][0]
    assert called_page.url == "https://cnaps.mg/page"
    assert called_page.classes == ["content-article"]
    assert count == 2


@patch("src.ingestion.transform.service._parse_html")
def test_transform_url_returns_zero_when_parse_fails(mock_parse):
    mock_parse.return_value = None

    from src.ingestion.transform.service import transform_url
    count = transform_url("https://cnaps.mg/page", [], "rag_cnaps")

    assert count == 0
