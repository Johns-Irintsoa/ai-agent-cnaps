import logging
import traceback

from fastapi import APIRouter, HTTPException

from ..schemas import ChatRequest, RAGResponse
from ...inference.service import ask_question

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


@router.post("/ask", response_model=RAGResponse)
async def ask(request: ChatRequest):
    try:
        result = await ask_question(
            request.message,
            matricule=request.matricule,
            password=request.password,
        )
        return RAGResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la requête: {str(e)}")
