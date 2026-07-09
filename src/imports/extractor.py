"""
Extractor — infraestrutura comum de extração estruturada via LLM

Duas responsabilidades, ambas reaproveitadas pelos três importadores:

  1. PDF → Markdown (mesmo pymupdf4llm do pipeline RAG, ver
     src/etl/pdf_converter.py — aqui a partir de bytes, sem passar pelo
     diretório de documentos normativos).
  2. Seção de markdown → objetos tipados, via Structured Outputs da OpenAI
     (`chat.completions.parse` com schema Pydantic). As seções são
     independentes, então as chamadas rodam em paralelo com ThreadPoolExecutor.

Falha em uma seção não derruba a extração inteira: vira um warning e as
demais seções seguem — o admin vê o problema no preview.
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import pymupdf
import pymupdf4llm
from pydantic import BaseModel

from src.agent.orchestrator import get_openai_client
from src.config import LLM_MODEL

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 4

T = TypeVar("T", bound=BaseModel)


def pdf_bytes_to_markdown(file_bytes: bytes) -> str:
    """Converte um PDF (bytes) para Markdown estruturado."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        return pymupdf4llm.to_markdown(tmp.name)


def pdf_bytes_to_text(file_bytes: bytes) -> str:
    """
    Extrai o texto puro do PDF (camada de texto, em ordem de leitura).

    Usado no lugar do Markdown quando a detecção de tabelas do pymupdf4llm
    atrapalha em vez de ajudar — no PDF do itinerário do transporte ela
    descarta tabelas inteiras (ex.: ônibus H das 16:10) e duplica cada linha
    em duas células, enquanto o texto puro preserva tudo.
    """
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def parse_structured(system_prompt: str, user_content: str, response_model: type[T]) -> T:
    """Uma chamada de Structured Outputs — retorna a instância parseada."""
    client = get_openai_client()
    completion = client.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=response_model,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM não retornou um objeto estruturado válido (refusal ou truncado)")
    return parsed


def parse_sections(
    system_prompt: str,
    sections: list[tuple[str, str]],
    response_model: type[T],
) -> tuple[list[tuple[str, T]], list[str]]:
    """
    Extrai várias seções em paralelo.

    Args:
        sections: lista de (rótulo, conteúdo) — o rótulo identifica a seção
            nos warnings (ex: "JANEIRO/2026", "ÔNIBUS A (manhã)").

    Returns:
        (resultados como (rótulo, objeto), warnings de seções que falharam)
    """
    results: list[tuple[str, T]] = []
    warnings: list[str] = []

    def _one(label: str, content: str) -> tuple[str, T | None, str | None]:
        try:
            return label, parse_structured(system_prompt, content, response_model), None
        except Exception as e:
            logger.warning(f"Extração falhou na seção '{label}': {e}")
            return label, None, f"Seção '{label}': extração falhou ({e})"

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = [pool.submit(_one, label, content) for label, content in sections]
        for future in futures:
            label, parsed, error = future.result()
            if parsed is not None:
                results.append((label, parsed))
            if error:
                warnings.append(error)

    return results, warnings


def save_markdown_artifact(md: str, job_dir: Path) -> None:
    """Guarda o markdown intermediário ao lado do PDF, para auditoria/debug."""
    try:
        (job_dir / "extracted.md").write_text(md, encoding="utf-8")
    except OSError as e:
        logger.warning(f"Não foi possível salvar artefato markdown em {job_dir}: {e}")
