"""
Courses Router — Consulta Pública de Cursos

Único endpoint público (protegido pela x-api-key, mesmo padrão de
/professors, /academic-events) — permite que consumidores do chat
descubram quais `course_id` são válidos para escopar `POST /chat`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.admin.schemas import CourseOut
from src.courses import service as course_service
from src.db.session import get_db

router = APIRouter()


@router.get("", response_model=list[CourseOut], summary="Lista os cursos")
async def list_courses(
    active_only: bool = False,
    db: Session = Depends(get_db),
) -> list[CourseOut]:
    return course_service.list_courses(db, active_only=active_only)
