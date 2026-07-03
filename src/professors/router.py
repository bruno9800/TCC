"""
Professors Router — Consulta Pública do Corpo Docente

Único endpoint público (protegido pela x-api-key, mesmo padrão de /chat,
/documents e /logs — não pela auth de admin, que fica em /admin/professors).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.admin.schemas import ProfessorOut
from src.db.session import get_db
from src.professors import service as professor_service

router = APIRouter()


@router.get("", response_model=list[ProfessorOut], summary="Lista o corpo docente")
async def list_professors(
    course_id: int | None = None,
    area: str | None = None,
    name: str | None = None,
    db: Session = Depends(get_db),
) -> list[ProfessorOut]:
    return professor_service.list_professors(db, course_id=course_id, area=area, name=name)
