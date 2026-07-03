"""
CourseService — CRUD mínimo sobre Course

Suporte à Fase 6 (escopo por curso): permite que o admin cadastre novos
cursos e que consumidores do chat descubram quais `course_id` são válidos
(GET /courses). PATCH/DELETE ficam fora de escopo — cursos mudam raramente,
mesma decisão já tomada para Discipline na Fase 3.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.db.models import Course


def create_course(db: Session, code: str, name: str, active: bool = True) -> Course:
    course = Course(code=code, name=name, active=active)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


def list_courses(db: Session, active_only: bool = False) -> list[Course]:
    query = db.query(Course)
    if active_only:
        query = query.filter(Course.active.is_(True))
    return query.order_by(Course.name).all()


def get_course(db: Session, course_id: int) -> Course | None:
    return db.get(Course, course_id)
