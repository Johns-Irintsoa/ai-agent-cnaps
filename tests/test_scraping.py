from ingestion.scraping.WebPageScrapper import get_urls_from_json, get_all_urls
from ingestion.scraping.models import WebPageContent


class TestGetUrlsFromJson:
    def test_loads_entries_from_custom_file(self, minimal_json_file):
        pages = get_urls_from_json(minimal_json_file)
        assert len(pages) == 2
        assert pages[0].url == "https://example.com/a"
        assert pages[0].classes == ["cls-a"]
        assert pages[1].is_paginated is False

    def test_returns_webpagecontent_objects(self, minimal_json_file):
        pages = get_urls_from_json(minimal_json_file)
        assert all(isinstance(p, WebPageContent) for p in pages)

    def test_loads_real_cnaps_json(self):
        pages = get_urls_from_json()
        assert len(pages) == 14
        urls = [p.url for p in pages]
        assert "https://www.cnaps.mg/fr/faq" in urls


CNAPS_HISTORIQUE_URL = "https://www.cnaps.mg/fr/cnaps/historique"
CNAPS_MISSIONS_URL = "https://www.cnaps.mg/fr/cnaps/missions-et-engagements"


class TestGetAllUrls:
    def test_non_paginated_pages_pass_through(self):
        pages = [
            WebPageContent(url=CNAPS_HISTORIQUE_URL),
            WebPageContent(url=CNAPS_MISSIONS_URL),
        ]
        result = get_all_urls(pages)
        assert result == [CNAPS_HISTORIQUE_URL, CNAPS_MISSIONS_URL]

    def test_empty_list_returns_empty(self):
        assert get_all_urls([]) == []
