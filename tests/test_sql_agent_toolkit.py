from unittest.mock import MagicMock
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from src.inference.SQLAgent.agent_toolkit import get_toolkit


def test_get_toolkit_returns_sqldatabasetoolkit():
    mock_db = MagicMock()
    mock_llm = MagicMock()

    toolkit = get_toolkit(mock_db, mock_llm)

    assert isinstance(toolkit, SQLDatabaseToolkit)


def test_toolkit_receives_correct_db_and_llm():
    mock_db = MagicMock()
    mock_llm = MagicMock()

    toolkit = get_toolkit(mock_db, mock_llm)

    assert toolkit.db is mock_db
    assert toolkit.llm is mock_llm
