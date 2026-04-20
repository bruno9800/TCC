"""
Query Logger

Registra cada interação com o sistema em data/logs/queries.jsonl.
Um objeto JSON por linha — fácil de ler, analisar e importar (pandas, etc.).

Campos registrados por entrada:
  - timestamp       : ISO 8601
  - question        : pergunta original do usuário
  - search_query    : query otimizada pelo agente para busca (se used_search)
  - used_search     : se o pipeline RAG foi acionado
  - sources         : documentos retornados com score e artigo
  - top_k           : parâmetro usado na requisição
  - filter_revoked  : parâmetro usado na requisição
  - tokens_prompt   : tokens de entrada consumidos
  - tokens_completion: tokens de saída consumidos
  - model           : modelo LLM usado
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOGS_DIR

logger = logging.getLogger(__name__)

_LOG_FILE: Path = LOGS_DIR / "queries.jsonl"


def log_query(
    question: str,
    search_query: str,
    used_search: bool,
    sources: list[dict],
    top_k: int,
    filter_revoked: bool,
    tokens_prompt: int,
    tokens_completion: int,
    model: str,
) -> None:
    """
    Persiste uma entrada de log no arquivo JSONL.

    Falhas de escrita são silenciadas — o log nunca deve derrubar a API.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "search_query": search_query if used_search else None,
        "used_search": used_search,
        "sources": sources,
        "top_k": top_k,
        "filter_revoked": filter_revoked,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "model": model,
    }

    try:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Falha ao registrar query no log: {e}")
