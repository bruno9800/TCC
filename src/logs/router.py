"""
Logs Router — Leitura dos registros de queries

Expõe os logs de interação para análise sem precisar acessar o servidor.
Protegido pela mesma API key do restante da aplicação.
"""

from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Query

from src.config import LOGS_DIR

router = APIRouter()

_LOG_FILE = LOGS_DIR / "queries.jsonl"


@router.get(
    "/queries",
    summary="Lista as queries registradas",
    description="Retorna as últimas N entradas do log de interações.",
)
async def get_queries(
    limit: int = Query(default=50, ge=1, le=1000, description="Número de entradas a retornar (mais recentes primeiro)"),
    only_search: bool = Query(default=False, description="Se true, retorna apenas queries que acionaram o pipeline RAG"),
) -> list[dict]:
    if not _LOG_FILE.exists():
        return []

    try:
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo de log: {e}")

    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if only_search and not entry.get("used_search"):
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        except json.JSONDecodeError:
            continue

    return entries


@router.get(
    "/queries/stats",
    summary="Estatísticas agregadas das queries",
)
async def get_stats() -> dict:
    """
    Retorna métricas agregadas úteis para análise do TCC:
    - total de queries
    - % que acionaram busca
    - documentos mais citados
    - total de tokens consumidos
    """
    if not _LOG_FILE.exists():
        return {"total": 0}

    try:
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo de log: {e}")

    total = 0
    total_search = 0
    total_tokens_prompt = 0
    total_tokens_completion = 0
    source_counts: dict[str, int] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        if entry.get("used_search"):
            total_search += 1
        total_tokens_prompt += entry.get("tokens_prompt", 0)
        total_tokens_completion += entry.get("tokens_completion", 0)

        for src in entry.get("sources", []):
            name = src.get("source", "unknown")
            source_counts[name] = source_counts.get(name, 0) + 1

    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_queries": total,
        "search_triggered": total_search,
        "search_rate": round(total_search / total, 3) if total else 0,
        "total_tokens_prompt": total_tokens_prompt,
        "total_tokens_completion": total_tokens_completion,
        "top_10_sources": [{"source": s, "count": c} for s, c in top_sources],
    }
