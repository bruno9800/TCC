"""
CalendarEventService — CRUD sobre AcademicEvent

Mesma lógica de ProfessorService (fecha D9): prazos do calendário acadêmico
são dado estruturado, consultado via SQL — não via RAG. Ver PLANO_V2.md,
Seção 5, e a decisão de puxar a Fase 7 do roadmap original para a Fase 5b.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.models import AcademicEvent


def create_event(
    db: Session,
    title: str,
    start_date: date,
    end_date: date | None = None,
    course_id: int | None = None,
    category: str | None = None,
    legal_reference: str | None = None,
    campus: str | None = None,
    academic_period: str | None = None,
) -> AcademicEvent:
    event = AcademicEvent(
        title=title,
        start_date=start_date,
        end_date=end_date,
        course_id=course_id,
        category=category,
        legal_reference=legal_reference,
        campus=campus,
        academic_period=academic_period,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_events(
    db: Session,
    course_id: int | None = None,
    category: str | None = None,
    academic_period: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[AcademicEvent]:
    query = db.query(AcademicEvent)
    if course_id is not None:
        # Inclui eventos institucionais (course_id IS NULL — a maioria do calendário
        # acadêmico real, que vale pra UNIVASF inteira), não só os do curso pedido.
        query = query.filter(
            or_(AcademicEvent.course_id == course_id, AcademicEvent.course_id.is_(None))
        )
    if category is not None:
        query = query.filter(AcademicEvent.category.ilike(f"%{category}%"))
    if academic_period is not None:
        query = query.filter(AcademicEvent.academic_period == academic_period)
    if date_from is not None:
        query = query.filter(
            (AcademicEvent.end_date >= date_from) | (AcademicEvent.start_date >= date_from)
        )
    if date_to is not None:
        query = query.filter(AcademicEvent.start_date <= date_to)
    return query.order_by(AcademicEvent.start_date).all()


def get_event(db: Session, event_id: int) -> AcademicEvent | None:
    return db.get(AcademicEvent, event_id)


def update_event(db: Session, event_id: int, **fields) -> AcademicEvent:
    """Atualiza os campos fornecidos (o router deve passar `exclude_unset=True`)."""
    event = db.get(AcademicEvent, event_id)
    if event is None:
        raise ValueError(f"AcademicEvent {event_id} não encontrado")

    for key, value in fields.items():
        setattr(event, key, value)

    db.commit()
    db.refresh(event)
    return event


def delete_event(db: Session, event_id: int) -> None:
    event = db.get(AcademicEvent, event_id)
    if event is None:
        raise ValueError(f"AcademicEvent {event_id} não encontrado")
    db.delete(event)
    db.commit()
