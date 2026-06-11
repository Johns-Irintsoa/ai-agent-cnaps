from unittest.mock import MagicMock, patch


def make_mock_collection(ids=None):
    col = MagicMock()
    col.get.return_value = {"ids": ids or []}
    return col


@patch("src.ingestion.store.delete.ChromaClient")
def test_delete_by_source_url_returns_count(mock_client_cls):
    mock_col = make_mock_collection(ids=["https://cnaps.mg/page#chunk0", "https://cnaps.mg/page#chunk1"])
    mock_client_cls.get_client.return_value.get_or_create_collection.return_value = mock_col

    from src.ingestion.store.delete import delete_by_source_url
    count = delete_by_source_url("https://cnaps.mg/page", "rag_cnaps")

    mock_col.delete.assert_called_once_with(where={"source_url": "https://cnaps.mg/page"})
    assert count == 2


@patch("src.ingestion.store.delete.ChromaClient")
def test_delete_by_source_url_not_found_returns_zero(mock_client_cls):
    mock_col = make_mock_collection()
    mock_client_cls.get_client.return_value.get_or_create_collection.return_value = mock_col

    from src.ingestion.store.delete import delete_by_source_url
    count = delete_by_source_url("https://cnaps.mg/inexistant", "rag_cnaps")

    mock_col.delete.assert_not_called()
    assert count == 0


@patch("src.ingestion.store.delete.ChromaClient")
def test_delete_by_ids(mock_client_cls):
    mock_col = make_mock_collection()
    mock_client_cls.get_client.return_value.get_or_create_collection.return_value = mock_col

    from src.ingestion.store.delete import delete_by_ids
    count = delete_by_ids(["id1", "id2"], "rag_cnaps")

    mock_col.delete.assert_called_once_with(ids=["id1", "id2"])
    assert count == 2


@patch("src.ingestion.store.delete.ChromaClient")
def test_delete_all(mock_client_cls):
    mock_col = make_mock_collection(ids=["id1", "id2", "id3"])
    mock_client_cls.get_client.return_value.get_or_create_collection.return_value = mock_col

    from src.ingestion.store.delete import delete_all
    count = delete_all("rag_cnaps")

    mock_col.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
    assert count == 3
