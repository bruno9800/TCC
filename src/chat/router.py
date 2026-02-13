"""
Chat Router — Endpoint REST para o Chat

Define o endpoint POST /chat que recebe a mensagem do usuário,
delega ao serviço de chat (agente leve) e retorna a resposta.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.chat.schemas import ChatRequest, ChatResponse
from src.chat.service import run_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Endpoint principal do chat.

    Recebe uma mensagem e o histórico de conversa, e retorna a resposta
    do agente com fontes e métricas de tokens.
    """
    try:
        response = run_chat(
            message=request.message,
            history=request.history,
            top_k=request.top_k,
            filter_revoked=request.filter_revoked,
        )
        return response
    except Exception as e:
        logger.error(f"Erro no chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
