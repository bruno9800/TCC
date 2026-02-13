#!/usr/bin/env python3
"""
Script para executar o pipeline ETL completo.

Uso:
    python scripts/run_etl.py

Converte todos os PDFs de regimentos_estatutos_resolucoes/ para Markdown
e aplica chunking semântico-hierárquico.
"""

import logging
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_DIR, CHUNKS_DIR
from src.etl.pdf_converter import run_etl
from src.etl.revocation_filter import analyze_revocation
from src.chunking.legal_chunker import chunk_document, save_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("PIPELINE ETL — Documentos Normativos UNIVASF")
    logger.info("=" * 60)

    # Fase 1: Conversão PDF → Markdown
    logger.info("\n📄 Fase 1: Conversão PDF → Markdown")
    documents = run_etl()

    if not documents:
        logger.error("Nenhum documento convertido. Verifique o diretório de PDFs.")
        return

    # Fase 2: Análise de revogação + Chunking
    logger.info("\n✂️  Fase 2: Chunking Semântico-Hierárquico")

    total_chunks = 0
    revoked_count = 0

    for doc in documents:
        # Verifica revogação
        revocation = analyze_revocation(doc.source_path, doc.markdown_text)

        if revocation.is_revoked:
            revoked_count += 1

        # Chunking
        chunks = chunk_document(
            markdown_text=doc.markdown_text,
            source=doc.filename,
            category=doc.category,
            status=revocation.status,
        )

        # Salva chunks em JSONL
        output_name = doc.filename.replace(" ", "_")
        save_chunks(chunks, output_name)
        total_chunks += len(chunks)

    # Resumo final
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DO ETL:")
    logger.info(f"  📄 Documentos processados: {len(documents)}")
    logger.info(f"  ⚠️  Documentos revogados: {revoked_count}")
    logger.info(f"  ✂️  Chunks gerados: {total_chunks}")
    logger.info(f"  📁 Chunks salvos em: {CHUNKS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
