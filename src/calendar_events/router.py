"""
Calendar Events Router — Consulta Pública do Calendário Acadêmico

Único endpoint público (protegido pela x-api-key, mesmo padrão de
/chat, /documents, /logs e /professors — não pela auth de admin, que fica
em /admin/academic-events).
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.admin.schemas import AcademicEventOut
from src.calendar_events import service as calendar_service
from src.db.session import get_db

router = APIRouter()


@router.get("", response_model=list[AcademicEventOut], summary="Lista eventos do calendário acadêmico")
async def list_events(
    course_id: int | None = None,
    category: str | None = None,
    academic_period: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: Session = Depends(get_db),
) -> list[AcademicEventOut]:
    return calendar_service.list_events(
        db,
        course_id=course_id,
        category=category,
        academic_period=academic_period,
        date_from=date_from,
        date_to=date_to,
    )
