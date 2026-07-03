"""
Chat Service — Adaptadores para o AgentOrchestrator (Fase 4)

run_chat/stream_chat são wrappers finos sobre src.agent.orchestrator.run(),
que faz toda a orquestração (decisão via function calling nativo, execução
de tools, síntese final). Isso resolve dois problemas que existiam aqui
antes da Fase 4:

  - D5: a decisão "buscar ou não" era um JSON parseado manualmente da
    resposta do LLM — agora é function calling nativo, no orchestrator.
  - D4: run_chat e stream_chat duplicavam quase toda a lógica — agora ambos
    só traduzem os eventos do mesmo generator para seus formatos de saída.

A sessão do banco é aberta e fechada aqui (não via Depends(get_db) no
router) porque com StreamingResponse o generator só é consumido pelo
Starlette depois que a função da rota já retornou — uma sessão injetada via
Depends seria fechada antes do generator (que usa `db` para ProfessorTool)
terminar de rodar.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from collections.abc import Generator

from src.agent import orchestrator
from src.chat.schemas import ChatMessage, ChatResponse, SourceInfo, TokenUsage
from src.config import LLM_MODEL
from src.db.session import SessionLocal
from src.logs.query_logger import log_query

logger = logging.getLogger(__name__)


def _build_history_messages(history: list[ChatMessage]) -> list[dict]:
    """Converte o histórico de ChatMessage para o formato da API OpenAI."""
    return [{"role": msg.role, "content": msg.content} for msg in history]


def _build_source_infos(raw_sources: list[dict]) -> list[SourceInfo]:
    """Traduz os dicts de `sources` retornados pelas tools para SourceInfo."""
    infos: list[SourceInfo] = []
    for src in raw_sources:
        if src.get("origin") == "professor":
            if src.get("nde_role"):
                nde_tag = f" (NDE — {src['nde_role']})"
            elif src.get("is_nde"):
                nde_tag = " (NDE)"
            else:
                nde_tag = ""
            infos.append(
                SourceInfo(
                    origin="professor",
                    source=src.get("name", ""),
                    category=f"Professor{nde_tag}",
                    snippet=f"{src.get('email', '')} — {src.get('area') or 'área não informada'}",
                )
            )
        else:
            source_name = src.get("source", "")
            infos.append(
                SourceInfo(
                    origin="rag",
                    source=source_name,
                    category=src.get("category", ""),
                    article_id=src.get("article_id", ""),
                    hierarchy=src.get("hierarchy", ""),
                    score=src.get("score", 0.0),
                    snippet=src.get("snippet", ""),
                    download_url=(
                        f"/documents/download?source={urllib.parse.quote(source_name)}"
                        if source_name
                        else ""
                    ),
                )
            )
    return infos


def _log(message: str, source_infos: list[SourceInfo], used_tools: list[str], tokens: dict,
          top_k: int, filter_revoked: bool, model: str) -> None:
    log_query(
        question=message,
        search_query="",
        used_search=bool(used_tools),
        sources=[{"source": s.source, "score": s.score, "article_id": s.article_id} for s in source_infos],
        top_k=top_k,
        filter_revoked=filter_revoked,
        tokens_prompt=tokens.get("prompt", 0),
        tokens_completion=tokens.get("completion", 0),
        model=model,
    )


def run_chat(
    message: str,
    history: list[ChatMessage],
    top_k: int = 5,
    filter_revoked: bool = True,
    model: str = LLM_MODEL,
) -> ChatResponse:
    """Executa o agente e retorna a resposta completa (não-streaming)."""
    db = SessionLocal()
    try:
        answer_parts: list[str] = []
        raw_sources: list[dict] = []
        used_tools: list[str] = []
        tokens = {"prompt": 0, "completion": 0}

        for event in orchestrator.run(
            message=message,
            history=_build_history_messages(history),
            db=db,
            top_k=top_k,
            filter_revoked=filter_revoked,
            model=model,
        ):
            if event["type"] == "token":
                answer_parts.append(event["text"])
            elif event["type"] == "done":
                raw_sources = event["sources"]
                used_tools = event["used_tools"]
                tokens = event["tokens"]
            elif event["type"] == "error":
                raise RuntimeError(event["text"])

        source_infos = _build_source_infos(raw_sources)
        _log(message, source_infos, used_tools, tokens, top_k, filter_revoked, model)

        return ChatResponse(
            answer="".join(answer_parts),
            sources=source_infos,
            model=model,
            tokens=TokenUsage(**tokens),
            used_search=bool(used_tools),
            used_tools=used_tools,
        )
    finally:
        db.close()


def stream_chat(
    message: str,
    history: list[ChatMessage],
    top_k: int = 5,
    filter_revoked: bool = True,
    model: str = LLM_MODEL,
) -> Generator[str, None, None]:
    """
    Versão streaming do run_chat.

    Emite eventos SSE:
      - data: {"type": "status", "text": "..."}   — etapas do pipeline
      - data: {"type": "token",  "text": "..."}   — tokens da resposta final
      - data: {"type": "done",   "sources": [...], "used_search": ..., "used_tools": [...]}
      - data: {"type": "error",  "text": "..."}   — erro
    """
    db = SessionLocal()

    def emit(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    try:
        for event in orchestrator.run(
            message=message,
            history=_build_history_messages(history),
            db=db,
            top_k=top_k,
            filter_revoked=filter_revoked,
            model=model,
        ):
            if event["type"] == "status":
                yield emit({"type": "status", "text": event["text"]})
            elif event["type"] == "token":
                yield emit({"type": "token", "text": event["text"]})
            elif event["type"] == "error":
                yield emit({"type": "error", "text": event["text"]})
            elif event["type"] == "done":
                source_infos = _build_source_infos(event["sources"])
                used_tools = event["used_tools"]
                yield emit(
                    {
                        "type": "done",
                        "sources": [s.model_dump() for s in source_infos],
                        "used_search": bool(used_tools),
                        "used_tools": used_tools,
                    }
                )
                _log(message, source_infos, used_tools, event["tokens"], top_k, filter_revoked, model)
    finally:
        db.close()
