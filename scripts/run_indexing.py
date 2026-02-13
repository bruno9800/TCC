#!/usr/bin/env python3
"""
Script para indexar chunks no ChromaDB.

Uso:
    python scripts/run_indexing.py

Carrega todos os chunks JSONL e cria embeddings + índice vetorial.
Requer OPENAI_API_KEY configurada no .env
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.legal_chunker import load_all_chunks
from src.indexing.vector_store import index_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("INDEXAÇÃO — ChromaDB + OpenAI Embeddings")
    logger.info("=" * 60)

    # Carrega todos os chunks
    logger.info("\n📦 Carregando chunks...")
    chunks = load_all_chunks()

    if not chunks:
        logger.error("Nenhum chunk encontrado. Execute o ETL primeiro:")
        logger.error("  python scripts/run_etl.py")
        return

    logger.info(f"  {len(chunks)} chunks carregados")

    # Filtra chunks vigentes e revogados separadamente para estatísticas
    vigentes = [c for c in chunks if c.metadata.status == "vigente"]
    revogados = [c for c in chunks if c.metadata.status == "revogado"]
    logger.info(f"  Vigentes: {len(vigentes)} | Revogados: {len(revogados)}")

    # Indexa todos (revogados são indexados mas filtrados na busca)
    logger.info("\n🔗 Gerando embeddings e indexando...")
    collection = index_chunks(chunks)

    logger.info("\n" + "=" * 60)
    logger.info("INDEXAÇÃO COMPLETA:")
    logger.info(f"  Total na collection: {collection.count()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
