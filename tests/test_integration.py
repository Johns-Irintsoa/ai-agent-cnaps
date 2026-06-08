import pytest

pytestmark = pytest.mark.integration


class TestParseHtml:
    def test_extracts_content_from_cnaps_page(self):
        from ingestion.transform.parsing import _parse_html
        result = _parse_html("https://www.cnaps.mg/fr/cnaps/historique")
        assert result is not None
        assert len(result.contenu_md) > 100
        assert result.metadata.source_url == "https://www.cnaps.mg/fr/cnaps/historique"

    def test_returns_none_for_invalid_url(self):
        from ingestion.transform.parsing import _parse_html
        result = _parse_html("https://url-qui-nexiste-pas.invalid")
        assert result is None


class TestExtractUrls:
    def test_returns_non_empty_list(self):
        from ingestion.scraping.WebPageScrapper import extract_urls
        results = extract_urls()
        assert len(results) > 0

    def test_all_results_have_content(self):
        from ingestion.scraping.WebPageScrapper import extract_urls
        results = extract_urls()
        assert all(len(r.contenu_md) > 0 for r in results)
