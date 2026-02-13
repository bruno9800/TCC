from __future__ import annotations

"""
PDF → Markdown Converter

Converte PDFs de documentos normativos da UNIVASF para Markdown estruturado,
preservando cabeçalhos, tabelas e hierarquia visual usando pymupdf4llm.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import pymupdf4llm

from src.config import DOCUMENTS_DIR, MARKDOWN_DIR

logger = logging.getLogger(__name__)


class ConvertedDocument(NamedTuple):
    """Resultado da conversão de um PDF."""

    source_path: Path
    output_path: Path
    category: str
    filename: str
    markdown_text: str


def classify_document(pdf_path: Path) -> str:
    """
    Classifica o tipo de documento com base no caminho do diretório.

    Returns:
        Uma das categorias: 'Estatuto', 'Regimento Geral',
        'Resolução PROEN', 'Resolução PROEX', 'Resolução PRPPGI'
    """
    relative = pdf_path.relative_to(DOCUMENTS_DIR)
    parts = relative.parts

    if len(parts) > 1:
        parent = parts[0].upper()
        if "PROEN" in parent:
            return "Resolução PROEN"
        elif "PROEX" in parent:
            return "Resolução PROEX"
        elif "PRPPGI" in parent:
            return "Resolução PRPPGI"

    name_lower = pdf_path.stem.lower()
    if "estatuto" in name_lower:
        return "Estatuto"
    elif "regimento" in name_lower:
        return "Regimento Geral"
    else:
        return "Resolução"


def is_revoked_by_filename(pdf_path: Path) -> bool:
    """Detecta se o arquivo está marcado como revogado no nome."""
    name_upper = pdf_path.stem.upper()
    return "REVOGADA" in name_upper or "REVOGADO" in name_upper


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """
    Converte um único PDF para Markdown usando pymupdf4llm.

    Args:
        pdf_path: Caminho absoluto para o arquivo PDF.

    Returns:
        Texto em formato Markdown.
    """
    logger.info(f"Convertendo: {pdf_path.name}")
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        return md_text
    except Exception as e:
        logger.error(f"Erro ao converter {pdf_path.name}: {e}")
        raise


def process_single_pdf(pdf_path: Path) -> ConvertedDocument:
    """
    Processa um único PDF: converte para Markdown e salva.

    Returns:
        ConvertedDocument com os metadados e texto.
    """
    category = classify_document(pdf_path)

    # Cria subdiretório para a categoria
    category_dir = MARKDOWN_DIR / category.replace(" ", "_").lower()
    category_dir.mkdir(parents=True, exist_ok=True)

    # Converte
    md_text = convert_pdf_to_markdown(pdf_path)

    # Salva o arquivo .md
    output_name = pdf_path.stem + ".md"
    output_path = category_dir / output_name
    output_path.write_text(md_text, encoding="utf-8")

    logger.info(f"  → Salvo em: {output_path.relative_to(MARKDOWN_DIR)}")

    return ConvertedDocument(
        source_path=pdf_path,
        output_path=output_path,
        category=category,
        filename=pdf_path.stem,
        markdown_text=md_text,
    )


def discover_pdfs(base_dir: Path | None = None) -> list[Path]:
    """
    Descobre todos os PDFs no diretório de documentos.

    Args:
        base_dir: Diretório base para busca. Padrão: DOCUMENTS_DIR

    Returns:
        Lista de caminhos para arquivos PDF encontrados.
    """
    base = base_dir or DOCUMENTS_DIR
    pdfs = sorted(base.rglob("*.pdf"))
    logger.info(f"Encontrados {len(pdfs)} PDFs em {base}")
    return pdfs


def run_etl(base_dir: Path | None = None) -> list[ConvertedDocument]:
    """
    Executa o pipeline ETL completo: descobre PDFs, converte para Markdown.

    Args:
        base_dir: Diretório base para busca de PDFs.

    Returns:
        Lista de documentos convertidos.
    """
    pdfs = discover_pdfs(base_dir)
    results: list[ConvertedDocument] = []
    errors: list[tuple[Path, Exception]] = []

    for pdf_path in pdfs:
        try:
            doc = process_single_pdf(pdf_path)
            results.append(doc)
        except Exception as e:
            errors.append((pdf_path, e))
            logger.error(f"Falha ao processar {pdf_path.name}: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"ETL Completo: {len(results)} convertidos, {len(errors)} erros")
    if errors:
        for path, err in errors:
            logger.error(f"  ERRO: {path.name} → {err}")

    return results
