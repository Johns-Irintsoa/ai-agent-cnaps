import pytest
from unittest.mock import patch, MagicMock
from src.db.oracle_client import OracleClient


def setup_function():
    """Réinitialise le singleton avant chaque test."""
    OracleClient._engine = None
    OracleClient._db = None
    OracleClient._initialized = False


def test_get_engine_returns_engine_on_success():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.oracle_client.create_engine", return_value=mock_engine):
        engine = OracleClient.get_engine()

    assert engine is mock_engine


def test_get_engine_returns_none_on_connection_failure():
    with patch("src.db.oracle_client.create_engine", side_effect=Exception("Connection refused")):
        engine = OracleClient.get_engine()

    assert engine is None


def test_get_engine_is_singleton():
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: MagicMock()
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.db.oracle_client.create_engine", return_value=mock_engine) as mock_create:
        OracleClient.get_engine()
        OracleClient.get_engine()

    mock_create.assert_called_once()


def test_get_db_returns_none_when_engine_unavailable():
    OracleClient._initialized = True
    OracleClient._engine = None

    result = OracleClient.get_db()

    assert result is None


def test_get_db_returns_sqldatabase_when_engine_available():
    mock_engine = MagicMock()
    mock_db = MagicMock()
    OracleClient._initialized = True
    OracleClient._engine = mock_engine

    with patch("src.db.oracle_client.SQLDatabase", return_value=mock_db):
        result = OracleClient.get_db()

    assert result is mock_db


def test_healthcheck_returns_false_when_engine_unavailable():
    OracleClient._initialized = True
    OracleClient._engine = None

    assert OracleClient.healthcheck() is False


def test_healthcheck_returns_true_when_engine_available():
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: MagicMock()
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    OracleClient._initialized = True
    OracleClient._engine = mock_engine

    assert OracleClient.healthcheck() is True


def test_close_resets_singleton():
    OracleClient._engine = MagicMock()
    OracleClient._db = MagicMock()
    OracleClient._initialized = True

    OracleClient.close()

    assert OracleClient._engine is None
    assert OracleClient._db is None
    assert OracleClient._initialized is False
