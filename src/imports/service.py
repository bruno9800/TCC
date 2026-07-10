"""
ImportService — ciclo de vida das importações estruturadas (PDF → dados relacionais)

Duas etapas com validação humana (human-in-the-loop):

  1. create_import: salva o PDF, converte para Markdown, extrai os itens via
     LLM e calcula o diff contra o banco — mas NÃO escreve nas tabelas de
     destino. O resultado fica em ImportJob (status='preview') para o admin
     revisar.
  2. apply_import: aplica o payload em transação única (status='applied');
     ou discard_import descarta (status='discarded').

O apply de cada tipo é idempotente por construção (replace-by-scope /
upsert / replace-by-semester — ver docstrings dos importadores).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT, RAW_DOCS_DIR
from src.db.models import Course, ImportJob
from src.imports import calendar_importer, curriculum_importer, routes_importer
from src.imports.extractor import (
    pdf_bytes_to_markdown,
    pdf_bytes_to_text,
    save_markdown_artifact,
)

logger = logging.getLogger(__name__)

IMPORT_TYPES = ("calendar", "curriculum", "transport")
SEMESTER_FORMAT = r"^\d{4}\.[12]$"


def create_import(
    db: Session,
    import_type: str,
    file_bytes: bytes,
    filename: str,
    course_id: int | None = None,
    semester: str | None = None,
    admin_id: int | None = None,
) -> ImportJob:
    """Extrai o PDF e registra o preview. Falha de extração → status='failed'."""
    if import_type not in IMPORT_TYPES:
        raise ValueError(f"import_type deve ser um de {IMPORT_TYPES}")
    if import_type == "curriculum":
        if course_id is None:
            raise ValueError("Importação de PPC exige course_id")
        if db.get(Course, course_id) is None:
            raise ValueError(f"Curso {course_id} não encontrado")
    if import_type == "transport" and not semester:
        raise ValueError("Importação de itinerário exige semester (ex: '2026.1')")

    job = ImportJob(
        import_type=import_type,
        course_id=course_id if import_type == "curriculum" else None,
        semester=semester if import_type == "transport" else None,
        filename=filename,
        status="preview",
        created_by_id=admin_id,
    )
    db.add(job)
    db.flush()

    job_dir = RAW_DOCS_DIR / "imports" / str(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / filename).write_bytes(file_bytes)
    job.storage_path = str((job_dir / filename).relative_to(PROJECT_ROOT))
    db.commit()

    try:
        # itinerário usa texto puro: a detecção de tabelas do pymupdf4llm
        # descarta/duplica tabelas nesse PDF (ver routes_importer)
        if import_type == "transport":
            md = pdf_bytes_to_text(file_bytes)
        else:
            md = pdf_bytes_to_markdown(file_bytes)
        save_markdown_artifact(md, job_dir)

        if import_type == "calendar":
            items, warnings = calendar_importer.extract(md)
            stats = calendar_importer.diff(db, items) if items else {}
        elif import_type == "curriculum":
            items, warnings = curriculum_importer.extract(md)
            stats, diff_warnings = (
                curriculum_importer.diff(db, items, job.course_id) if items else ({}, [])
            )
            warnings = warnings + diff_warnings
        else:
            items, warnings = routes_importer.extract(md)
            stats = routes_importer.diff(db, items, job.semester) if items else {}

        job.payload = {"items": items}
        job.stats = stats
        job.warnings = warnings
        if not items:
            job.status = "failed"
            job.error_message = "Extração não produziu itens — ver warnings."
    except Exception as e:
        logger.error(f"Extração falhou para import_job={job.id}: {e}", exc_info=True)
        job.status = "failed"
        job.error_message = str(e)

    db.commit()
    db.refresh(job)
    return job


def list_imports(
    db: Session, import_type: str | None = None, status: str | None = None
) -> list[ImportJob]:
    query = db.query(ImportJob)
    if import_type is not None:
        query = query.filter(ImportJob.import_type == import_type)
    if status is not None:
        query = query.filter(ImportJob.status == status)
    return query.order_by(ImportJob.id.desc()).all()


def get_import(db: Session, job_id: int) -> ImportJob | None:
    return db.get(ImportJob, job_id)


def apply_import(db: Session, job_id: int) -> ImportJob:
    """Aplica o payload de um preview no banco, em transação única."""
    job = db.get(ImportJob, job_id)
    if job is None:
        raise ValueError(f"ImportJob {job_id} não encontrado")
    if job.status != "preview":
        raise ValueError(f"ImportJob {job_id} não está em preview (status='{job.status}')")

    items = (job.payload or {}).get("items", [])
    try:
        if job.import_type == "calendar":
            result = calendar_importer.apply(db, items)
        elif job.import_type == "curriculum":
            result = curriculum_importer.apply(db, items, job.course_id)
        else:
            result = routes_importer.apply(db, items, job.semester)

        job.status = "applied"
        job.applied_at = datetime.now()
        job.stats = {**(job.stats or {}), "apply": result}
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(job)
    return job


def discard_import(db: Session, job_id: int) -> None:
    job = db.get(ImportJob, job_id)
    if job is None:
        raise ValueError(f"ImportJob {job_id} não encontrado")
    if job.status == "applied":
        raise ValueError("Importação já aplicada não pode ser descartada")
    job.status = "discarded"
    db.commit()


def delete_import_file(job: ImportJob) -> None:
    """Remove o PDF salvo de um job descartado (housekeeping opcional)."""
    if job.storage_path:
        path = Path(PROJECT_ROOT) / job.storage_path
        path.unlink(missing_ok=True)
