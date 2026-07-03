"""
Ingestion Service — Pipeline reaproveitável de ETL → Chunking → Embedding → Indexação

Extrai a lógica que antes vivia espalhada em scripts/run_etl.py e
scripts/run_indexing.py para um serviço único, orientado a um `document_id`
já registrado no banco (ver src/db/models.py Document). Usado hoje pelos
scripts CLI (que viram wrappers finos) e, na Fase 2, pela API de upload.

Corrige dois problemas diagnosticados em PLANO_V2.md:
  - D1: o índice BM25 do HybridSearchEngine é um snapshot em memória — fica
    desatualizado após uma nova indexação até o processo reiniciar. Resolvido
    chamando `get_search_engine().reload()` ao final de embed_and_index().
  - D2: reindexar um documento só fazia `upsert` — chunks antigos que não
    existem mais na nova versão ficavam órfãos no ChromaDB. Resolvido
    apagando os `chroma_id`s antigos (via DocumentChunk, ou via filtro pelo
    metadado `source` para vetores legados da v1 sem DocumentChunk) antes de
    indexar os novos.

Também corrige a colisão de nomes encontrada na Fase 0 (dois documentos com
o mesmo filename sobrescrevendo o JSONL um do outro): os artefatos de
ingestão passam a ser nomeados por `document.id` (chave primária, única por
construção) em vez de por nome de arquivo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import CHUNKS_DIR, PROJECT_ROOT
from src.chunking.legal_chunker import chunk_document, load_chunks, save_chunks
from src.db.models import Document, DocumentChunk, IngestionJob
from src.etl.pdf_converter import convert_pdf_to_markdown
from src.etl.revocation_filter import analyze_revocation
from src.indexing.vector_store import get_or_create_collection, index_chunks
from src.retrieval.hybrid_search import get_search_engine

logger = logging.getLogger(__name__)


def _chunks_path(document_id: int) -> Path:
    """Caminho do JSONL de chunks de um documento — nomeado por ID, não por filename."""
    return CHUNKS_DIR / f"doc{document_id}.jsonl"


def etl_and_chunk(document_id: int, db: Session) -> None:
    """
    PDF → Markdown → chunks. Salva o resultado em data/chunks/doc{id}.jsonl
    e atualiza Document.status para "chunked" (ou "failed" em caso de erro).
    """
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} não encontrado")

    job = IngestionJob(
        document_id=document_id,
        stage="etl",
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)

    try:
        pdf_path = PROJECT_ROOT / document.storage_path
        markdown_text = convert_pdf_to_markdown(pdf_path)
        revocation = analyze_revocation(pdf_path, markdown_text)

        job.stage = "chunking"
        chunks = chunk_document(
            markdown_text=markdown_text,
            source=document.title,
            category=document.category or "",
            status=revocation.status,
            kb_slug=document.knowledge_base.slug,
            course_id=document.course_id,
        )
        save_chunks(chunks, f"doc{document.id}")

        document.revoked = revocation.is_revoked
        document.revoked_reason = (
            "Detectado automaticamente pelo nome do arquivo (revocation_filter)"
            if revocation.is_revoked
            else None
        )
        document.status = "chunked"
        job.status = "done"
    except Exception as e:
        document.status = "failed"
        job.status = "failed"
        job.error_message = str(e)[:2000]
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.error(f"Falha no ETL/chunking de document_id={document_id}: {e}")
        raise
    else:
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            f"ETL+chunking concluído para document_id={document_id} ({len(chunks)} chunks)"
        )


def embed_and_index(document_id: int, db: Session) -> None:
    """
    Carrega o JSONL gerado por etl_and_chunk, gera embeddings, indexa no
    ChromaDB, espelha os IDs em DocumentChunk, limpa vetores antigos (se
    for uma reindexação) e recarrega o índice BM25 do motor de busca.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} não encontrado")

    job = IngestionJob(
        document_id=document_id,
        stage="embedding",
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)

    try:
        jsonl_path = _chunks_path(document.id)
        chunks = load_chunks(jsonl_path)
        if not chunks:
            raise ValueError(
                f"Nenhum chunk encontrado em {jsonl_path} — rode etl_and_chunk primeiro"
            )

        collection = get_or_create_collection()

        # Limpa vetores antigos antes de reindexar — apenas os rastreados via
        # DocumentChunk (chroma_id preciso). Documentos sem DocumentChunk (ex:
        # os 48 importados pelo backfill da Fase 0, indexados pela v1 antes
        # desta tabela existir) NÃO disparam limpeza automática por metadado
        # `source` aqui: como esse campo não é garantidamente único (é
        # justamente a causa da colisão PROEN/PROEX da Fase 0), uma limpeza
        # automática por `source` arriscaria apagar vetores de OUTRO documento
        # que já tenha sido migrado e compartilhe o mesmo título. A migração
        # inicial de artefatos legados sem DocumentChunk é feita uma única vez,
        # manualmente e sob supervisão (ver EVOLUTION_V2.md, Fase 1) — depois
        # disso todo documento passa a ter DocumentChunk e cai no caminho
        # seguro acima.
        old_chunk_rows = db.query(DocumentChunk).filter_by(document_id=document.id).all()
        if old_chunk_rows:
            collection.delete(ids=[row.chroma_id for row in old_chunk_rows])
            for row in old_chunk_rows:
                db.delete(row)
            document.version += 1
            logger.info(
                f"  Removidos {len(old_chunk_rows)} chunks antigos (rastreados) "
                f"do document_id={document_id}"
            )

        job.stage = "indexing"
        chunk_ids = index_chunks(chunks, collection=collection, id_prefix=f"doc{document.id}")

        for chunk, chroma_id in zip(chunks, chunk_ids):
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chroma_id=chroma_id,
                    article_id=chunk.metadata.article_id,
                    hierarchy=" > ".join(chunk.metadata.hierarchy),
                    token_count=chunk.token_count(),
                )
            )

        document.status = "indexed"
        document.indexed_at = datetime.now(timezone.utc)
        job.status = "done"
    except Exception as e:
        document.status = "failed"
        job.status = "failed"
        job.error_message = str(e)[:2000]
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.error(f"Falha na indexação de document_id={document_id}: {e}")
        raise
    else:
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            f"Indexação concluída para document_id={document_id} ({len(chunk_ids)} vetores)"
        )
        get_search_engine().reload()


def process_document(document_id: int, db: Session) -> None:
    """Pipeline completo (ETL → chunk → embed → index) — usado pela futura API de upload."""
    etl_and_chunk(document_id, db)
    embed_and_index(document_id, db)
