"""
Chat Router — Endpoint REST para o Chat

Define o endpoint POST /chat que recebe a mensagem do usuário,
delega ao serviço de chat (agente leve) e retorna a resposta.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.chat.schemas import ChatRequest, ChatResponse
from src.chat.service import run_chat, stream_chat

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


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint de chat com streaming via Server-Sent Events (SSE).

    Emite eventos progressivos enquanto o pipeline executa:
      - status: etapas do processamento (busca, reranking, geração)
      - token:  tokens da resposta à medida que são gerados
      - done:   evento final com as fontes utilizadas
      - error:  em caso de falha

    Formato de cada evento: `data: {json}\\n\\n`
    """
    return StreamingResponse(
        stream_chat(
            message=request.message,
            history=request.history,
            top_k=request.top_k,
            filter_revoked=request.filter_revoked,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
