#!/usr/bin/env python3
"""
Script de Indexação — Embeddings + ChromaDB.

Itera os documentos registrados no banco com status "chunked" (já passaram
pelo ETL/chunking, prontos para embedding) e chama
IngestionService.embed_and_index() para cada um.

Sem argumentos, é seguro rodar a qualquer momento: documentos que já estão
"indexed" não são retocados. Requer OPENAI_API_KEY configurada no .env.

Uso:
    python scripts/run_indexing.py                  # indexa pendentes
    python scripts/run_indexing.py --reindex 34      # força reindexar um documento específico
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Document
from src.db.session import SessionLocal
from src.ingestion.service import embed_and_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    logger.info("INDEXAÇÃO — ChromaDB + OpenAI Embeddings")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        if args.reindex is not None:
            document_ids = [args.reindex]
        else:
            documents = db.query(Document).filter(Document.status == "chunked").all()
            document_ids = [d.id for d in documents]

        if not document_ids:
            logger.info("Nenhum documento pendente de indexação. Nada a fazer.")
            return

        logger.info(f"{len(document_ids)} documento(s) a indexar: {document_ids}")

        ok, failed = 0, 0
        for document_id in document_ids:
            try:
                embed_and_index(document_id, db)
                ok += 1
            except Exception as e:
                failed += 1
                logger.error(f"  Falha no document_id={document_id}: {e}")

        logger.info("\n" + "=" * 60)
        logger.info(f"INDEXAÇÃO COMPLETA: {ok} ok, {failed} falhas")
        logger.info("=" * 60)
    finally:
        db.close()


if __name__ == "__main__":
    main()
