from ingestion.transform.utils import (
    normalize_md_text,
    flatten_metadata,
    prefix_chunk_with_context,
    generate_chunk_id,
)
from ingestion.scraping.models import WebPageContentExtracted, WebPageMetadata


def _make_page(url="https://ex.com", title="Titre", date="2024-01-01", md="Contenu"):
    return WebPageContentExtracted(
        contenu_md=md,
        metadata=WebPageMetadata(source_url=url, title=title, date_posted=date),
    )


class TestNormalizeMdText:
    def test_collapse_triple_newlines(self):
        result = normalize_md_text("A\n\n\n\nB")
        assert "\n\n\n" not in result
        assert result == "A\n\nB"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_md_text("\n\n  Texte  \n\n") == "Texte"

    def test_single_newlines_untouched(self):
        assert normalize_md_text("A\nB") == "A\nB"


class TestFlattenMetadata:
    def test_returns_flat_dict_with_correct_keys(self):
        page = _make_page(url="https://cnaps.mg", title="Accueil", date="2024-06-01")
        result = flatten_metadata(page)
        assert result == {
            "source_url": "https://cnaps.mg",
            "title": "Accueil",
            "date_posted": "2024-06-01",
        }


class TestPrefixChunkWithContext:
    def test_prefix_contains_title_and_content(self):
        result = prefix_chunk_with_context("Corps du texte", "Mon document")
        assert result == "Document : Mon document\nContenu : Corps du texte"


class TestGenerateChunkId:
    def test_format_url_hash_chunk_n(self):
        assert generate_chunk_id("https://example.com/page", 0) == "https://example.com/page#chunk0"
        assert generate_chunk_id("https://example.com/page", 3) == "https://example.com/page#chunk3"
