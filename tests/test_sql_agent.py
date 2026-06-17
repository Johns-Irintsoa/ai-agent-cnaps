import pytest
from unittest.mock import patch, MagicMock

from src.inference.SQLAgent.SQL_agent import run_sql_agent


@pytest.mark.asyncio
async def test_run_sql_agent_returns_answer_when_db_available():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "La dernière période est 202412."}

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("Dernière période matricule 512196")

    assert result["answer"] == "La dernière période est 202412."
    assert result["needs_matricule"] is False
    assert result["metadata"] is None


@pytest.mark.asyncio
async def test_run_sql_agent_returns_unavailable_when_db_is_none():
    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=None):
        result = await run_sql_agent("512196")

    assert "indisponible" in result["answer"].lower()
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_run_sql_agent_handles_hors_perimetre():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "HORS_PERIMETRE"}

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("Quelque chose hors périmètre")

    assert result["answer"] == "Cette information n'est pas disponible dans la base de données CNaPS."
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_run_sql_agent_handles_exception_gracefully():
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = Exception("Oracle timeout")

    with patch("src.inference.SQLAgent.SQL_agent.OracleClient.get_db", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.get_toolkit", return_value=MagicMock()), \
         patch("src.inference.SQLAgent.SQL_agent.build_sql_agent", return_value=mock_agent), \
         patch("src.inference.SQLAgent.SQL_agent.LLMClient") as mock_llm_cls:
        mock_llm_cls.return_value.model = MagicMock()
        result = await run_sql_agent("512196")

    assert result["needs_matricule"] is False
    assert len(result["answer"]) > 0
