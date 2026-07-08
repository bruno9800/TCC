"""
DocumentService — CRUD administrativo sobre Document

Usado pelas rotas /admin/documents/* (src/admin/router.py). Reaproveita
IngestionService (src/ingestion/service.py, Fase 1) para o pipeline
ETL → chunk → embed → index; não reimplementa nada disso aqui.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import CHUNKS_DIR, PROJECT_ROOT, RAW_DOCS_DIR
from src.db.models import Document, DocumentChunk, IngestionJob, KnowledgeBase
from src.indexing.vector_store import get_or_create_collection
from src.ingestion.service import process_document
from src.retrieval.hybrid_search import get_search_engine

logger = logging.getLogger(__name__)


def create_document(
    db: Session,
    file_bytes: bytes,
    filename: str,
    knowledge_base_slug: str,
    course_id: int | None = None,
    title: str | None = None,
    category: str | None = None,
    admin_id: int | None = None,
) -> Document:
    """
    Registra um novo documento, salva o arquivo em data/raw/{id}/ e dispara a
    ingestão completa (ETL → chunk → embed → index).

    Se a ingestão falhar, a exceção é capturada e logada — o Document já fica
    com status="failed" e o erro registrado em IngestionJob (comportamento do
    próprio IngestionService), e esta função retorna o documento normalmente
    em vez de propagar um 500: o upload em si teve sucesso.
    """
    kb = db.query(KnowledgeBase).filter_by(slug=knowledge_base_slug).one_or_none()
    if kb is None:
        raise ValueError(f"Knowledge base '{knowledge_base_slug}' não encontrada")

    document = Document(
        knowledge_base_id=kb.id,
        course_id=course_id,
        title=title or Path(filename).stem,
        filename=filename,
        storage_path="",
        category=category,
        status="processing",
        uploaded_by_id=admin_id,
    )
    db.add(document)
    db.flush()  # obtém document.id sem commitar ainda

    dest = RAW_DOCS_DIR / str(document.id) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)

    document.storage_path = str(dest.relative_to(PROJECT_ROOT))
    document.checksum = hashlib.sha256(file_bytes).hexdigest()
    db.commit()
    db.refresh(document)

    try:
        process_document(document.id, db)
    except Exception as e:
        logger.warning(f"Ingestão falhou para document_id={document.id}: {e}")

    db.refresh(document)
    return document


def list_documents(
    db: Session,
    course_id: int | None = None,
    knowledge_base_id: int | None = None,
    status: str | None = None,
) -> list[Document]:
    query = db.query(Document)
    if course_id is not None:
        query = query.filter(Document.course_id == course_id)
    if knowledge_base_id is not None:
        query = query.filter(Document.knowledge_base_id == knowledge_base_id)
    if status is not None:
        query = query.filter(Document.status == status)
    return query.order_by(Document.id).all()


def get_document(db: Session, document_id: int) -> Document | None:
    return db.get(Document, document_id)


def update_document(db: Session, document_id: int, **fields) -> Document:
    """
    Atualiza os campos fornecidos. O chamador (router) deve passar apenas os
    campos explicitamente enviados no payload (ex: via
    `DocumentUpdateRequest.model_dump(exclude_unset=True)`), para que seja
    possível limpar um campo para null explicitamente sem confundir com
    "campo não informado".
    """
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} não encontrado")

    for key, value in fields.items():
        setattr(document, key, value)

    db.commit()
    db.refresh(document)
    return document


def reindex_document(db: Session, document_id: int) -> Document:
    """Força um novo ciclo ETL → chunk → embed → index para um documento existente."""
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} não encontrado")

    try:
        process_document(document_id, db)
    except Exception as e:
        logger.warning(f"Reindexação falhou para document_id={document_id}: {e}")

    db.refresh(document)
    return document


def delete_document(db: Session, document_id: int) -> None:
    """Remove o documento: vetores no ChromaDB, arquivo físico, JSONL e a linha no banco."""
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} não encontrado")

    chunk_rows = db.query(DocumentChunk).filter_by(document_id=document.id).all()
    if chunk_rows:
        collection = get_or_create_collection()
        collection.delete(ids=[row.chroma_id for row in chunk_rows])

    if document.storage_path:
        file_path = PROJECT_ROOT / document.storage_path
        file_path.unlink(missing_ok=True)
        try:
            file_path.parent.rmdir()  # remove data/raw/{id}/ se ficou vazio
        except OSError:
            pass  # não vazio (outros arquivos) ou já não existe — sem problema

    (CHUNKS_DIR / f"doc{document.id}.jsonl").unlink(missing_ok=True)

    # IngestionJob não tem relationship/cascade em Document (ao contrário de
    # DocumentChunk) — sem isso, o DELETE do Document falha por violação de FK.
    db.query(IngestionJob).filter_by(document_id=document.id).delete()

    db.delete(document)  # cascade remove DocumentChunk (relationship já configurada)
    db.commit()

    get_search_engine().reload()
