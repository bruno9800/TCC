#!/usr/bin/env python3
"""
Script de backfill — registra no banco relacional os documentos já
processados pela v1 (ETL + indexação manual via run_etl.py/run_indexing.py).

Cria uma linha `Document` para cada PDF em regimentos_estatutos_resolucoes/,
vinculada à knowledge base "regulamentos" (criada por seed_db.py), com
status="indexed" — já que esses documentos já estão no ChromaDB desde a v1.

Não cria `DocumentChunk` (mapeamento de chroma_id) — isso é Fase 1, quando o
IngestionService assumir a responsabilidade de reindexação e limpeza de
chunks órfãos.

Idempotente: pula PDFs cujo `storage_path` (caminho relativo, único mesmo
quando dois arquivos de pastas diferentes compartilham o mesmo nome — caso
real encontrado em PROEN/ e PROEX/) já tenha um Document registrado.

NOTA — achado durante o backfill: dois PDFs distintos (um em PROEN/, outro em
PROEX/) compartilham o mesmo stem
"resolucao-n-03-2022_curricularizao-da-extenso-na-univasf-pdf-nuvem-univasf".
Como `save_chunks()` em src/chunking/legal_chunker.py usa apenas o filename
(sem categoria) como nome do JSONL de saída, a segunda execução do ETL da v1
sobrescreveu o JSONL da primeira — ou seja, hoje só um dos dois documentos
tem chunks vivos no ChromaDB, embora ambos fiquem registrados aqui como
"indexed". Este script não corrige isso (fora do escopo da Fase 0); a
correção pertence à Fase 1 (IngestionService), que deve nomear o output por
document_id em vez de por filename.

Uso:
    python scripts/backfill_documents.py
"""

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_DIR, PROJECT_ROOT
from src.db.models import Document, KnowledgeBase
from src.db.session import SessionLocal
from src.etl.pdf_converter import classify_document, discover_pdfs, is_revoked_by_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    logger.info("=" * 60)
    logger.info("BACKFILL — Registro de documentos já processados")
    logger.info("=" * 60)

    session = SessionLocal()
    try:
        kb = session.query(KnowledgeBase).filter_by(slug="regulamentos").one_or_none()
        if kb is None:
            logger.error(
                "Knowledge base 'regulamentos' não encontrada. "
                "Rode `python scripts/seed_db.py` primeiro."
            )
            return

        pdfs = discover_pdfs(DOCUMENTS_DIR)
        created = 0
        skipped = 0

        for pdf_path in pdfs:
            filename = pdf_path.name
            storage_path = str(pdf_path.relative_to(PROJECT_ROOT))

            existing = session.query(Document).filter_by(storage_path=storage_path).one_or_none()
            if existing:
                skipped += 1
                continue

            category = classify_document(pdf_path)
            revoked = is_revoked_by_filename(pdf_path)

            doc = Document(
                knowledge_base_id=kb.id,
                course_id=None,  # institucional — aplica-se a todos os cursos
                title=pdf_path.stem,
                filename=filename,
                storage_path=storage_path,
                checksum=_checksum(pdf_path),
                category=category,
                status="indexed",
                version=1,
                revoked=revoked,
                revoked_reason="Detectado pelo nome do arquivo (ETL v1)" if revoked else None,
            )
            session.add(doc)
            created += 1
            logger.info(f"  + {filename} ({category}){' [REVOGADO]' if revoked else ''}")

        session.commit()
        logger.info("\n" + "=" * 60)
        logger.info(f"BACKFILL COMPLETO: {created} criados, {skipped} já existiam")
        logger.info("=" * 60)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
