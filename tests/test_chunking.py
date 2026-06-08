from ingestion.scraping.models import WebPageContentExtracted, WebPageMetadata
from ingestion.transform.splitting import chunking_md_html
from ingestion.transform.models import WebPageContentChunked


CNAPS_HISTORIQUE_URL = "https://www.cnaps.mg/fr/cnaps/historique"
CNAPS_FAQ_URL = "https://www.cnaps.mg/fr/faq"


def _long_page() -> WebPageContentExtracted:
    content = "\n\n".join([f"## Section {i}\n\nContenu section {i}." for i in range(30)])
    return WebPageContentExtracted(
        contenu_md=content,
        metadata=WebPageMetadata(
            source_url=CNAPS_FAQ_URL,
            title="FAQ CNaPS",
            date_posted="2024-01-01",
        ),
    )


class TestChunkingMdHtml:
    def test_returns_non_empty_list(self, sample_page):
        chunks = chunking_md_html(sample_page)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_first_chunk_id_starts_at_zero(self, sample_page):
        chunks = chunking_md_html(sample_page)
        assert chunks[0].id == f"{CNAPS_HISTORIQUE_URL}#chunk0"

    def test_chunk_ids_are_unique(self):
        chunks = chunking_md_html(_long_page())
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_document_prefixed_with_title(self, sample_page):
        chunks = chunking_md_html(sample_page)
        for chunk in chunks:
            assert chunk.document.startswith("Document : Historique CNaPS")

    def test_chunk_metadata_contains_source_url(self, sample_page):
        chunks = chunking_md_html(sample_page)
        for chunk in chunks:
            assert chunk.metadata["source_url"] == CNAPS_HISTORIQUE_URL

    def test_chunk_metadata_index_and_total_consistent(self, sample_page):
        chunks = chunking_md_html(sample_page)
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == total

    def test_returns_webpagecontentchunked_objects(self, sample_page):
        chunks = chunking_md_html(sample_page)
        assert all(isinstance(c, WebPageContentChunked) for c in chunks)
