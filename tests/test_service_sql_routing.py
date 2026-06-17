import pytest
from unittest.mock import patch, AsyncMock

from src.inference.service import _is_cotisation_intent, _extract_matricule


# --- Détection d'intention cotisation ---

def test_detects_cotisation_keyword():
    assert _is_cotisation_intent("Quelle est ma dernière période de cotisation ?") is True

def test_detects_cotise_keyword():
    assert _is_cotisation_intent("J'ai cotisé pendant combien de temps ?") is True

def test_detects_periode_keyword():
    assert _is_cotisation_intent("Quelle est ma dernière période ?") is True

def test_no_intent_for_unrelated_question():
    assert _is_cotisation_intent("Quels sont les droits à la retraite ?") is False

def test_no_intent_for_generic_question():
    assert _is_cotisation_intent("Comment contacter la CNaPS ?") is False


# --- Extraction de matricule ---

def test_extracts_matricule_alone():
    assert _extract_matricule("512196") == "512196"

def test_extracts_matricule_in_sentence():
    assert _extract_matricule("Mon matricule est 512196 merci") == "512196"

def test_extracts_7_digit_matricule():
    assert _extract_matricule("1234567") == "1234567"

def test_extracts_5_digit_matricule():
    assert _extract_matricule("12345") == "12345"

def test_no_matricule_for_8_digits():
    assert _extract_matricule("12345678") is None

def test_no_matricule_for_4_digits():
    assert _extract_matricule("1234") is None

def test_no_matricule_in_text_question():
    assert _extract_matricule("Quelle est ma dernière période de cotisation ?") is None


# --- Routing dans ask_question ---

@pytest.mark.asyncio
async def test_ask_question_routes_to_sql_agent_when_matricule_present():
    mock_result = {
        "answer": "Dernière période : 202412",
        "needs_matricule": False,
        "metadata": None,
        "evaluation": {},
        "from_cache": False,
    }
    with patch("src.inference.service.run_sql_agent", new_callable=AsyncMock, return_value=mock_result):
        from src.inference.service import ask_question
        result = await ask_question("512196")

    assert result["answer"] == "Dernière période : 202412"
    assert result["needs_matricule"] is False


@pytest.mark.asyncio
async def test_ask_question_returns_needs_matricule_when_intent_no_matricule():
    with patch("src.inference.service.run_sql_agent") as mock_sql:
        from src.inference.service import ask_question
        result = await ask_question("Quelle est ma dernière période de cotisation ?")

    mock_sql.assert_not_called()
    assert result["needs_matricule"] is True
    assert "matricule" in result["answer"].lower()
