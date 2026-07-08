#!/usr/bin/env python3
"""
Script de ETL — PDF → Markdown → Chunks.

Itera os documentos registrados no banco (ver Document em src/db/models.py)
com status "processing" ou "failed" (i.e. ainda não passaram pelo ETL/chunking
com sucesso) e chama IngestionService.etl_and_chunk() para cada um.

Sem argumentos, é seguro rodar a qualquer momento: documentos já
processados (status "chunked"/"indexed") não são retocados.

Uso:
    python scripts/run_etl.py                  # processa pendentes
    python scripts/run_etl.py --reindex 34      # força reprocessar um documento específico
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Document
from src.db.session import SessionLocal
from src.ingestion.service import etl_and_chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PENDING_STATUSES = ("processing", "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reindex",
        type=int,
        metavar="DOCUMENT_ID",
        help="Força o reprocessamento de um documento específico, independente do status atual.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ETL — PDF → Markdown → Chunks")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        if args.reindex is not None:
            document_ids = [args.reindex]
        else:
            documents = db.query(Document).filter(Document.status.in_(PENDING_STATUSES)).all()
            document_ids = [d.id for d in documents]

        if not document_ids:
            logger.info("Nenhum documento pendente de ETL. Nada a fazer.")
            return

        logger.info(f"{len(document_ids)} documento(s) a processar: {document_ids}")

        ok, failed = 0, 0
        for document_id in document_ids:
            try:
                etl_and_chunk(document_id, db)
                ok += 1
            except Exception as e:
                failed += 1
                logger.error(f"  Falha no document_id={document_id}: {e}")

        logger.info("\n" + "=" * 60)
        logger.info(f"ETL COMPLETO: {ok} ok, {failed} falhas")
        logger.info("=" * 60)
    finally:
        db.close()


if __name__ == "__main__":
    main()
