"""
TransportService — consulta sobre rotas do transporte estudantil (PROAE)

Mesma lógica de CalendarEventService (D9): horários exatos de ônibus são dado
estruturado, consultado via SQL — não via RAG. Populado exclusivamente pela
importação estruturada (src/imports/routes_importer.py); não há CRUD manual
de paradas — para corrigir uma rota, reimporta-se o PDF.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from src.db.models import TransportRoute, TransportRouteStop


def latest_semester(db: Session) -> str | None:
    """Semestre mais recente com rotas cadastradas (ordenação lexicográfica de 'AAAA.N')."""
    return db.query(func.max(TransportRoute.semester)).scalar()


def list_routes(
    db: Session,
    semester: str | None = None,
    shift: str | None = None,
    location: str | None = None,
) -> list[TransportRoute]:
    """
    Lista rotas com paradas (eager). `location` filtra rotas que passam por um
    ponto cujo nome contenha o termo (case-insensitive).
    """
    if semester is None:
        semester = latest_semester(db)
        if semester is None:
            return []

    query = (
        db.query(TransportRoute)
        .options(joinedload(TransportRoute.stops))
        .filter(TransportRoute.semester == semester)
    )
    if shift is not None:
        query = query.filter(TransportRoute.shift == shift.strip().lower())
    if location is not None:
        query = (
            query.join(TransportRoute.stops)
            .filter(TransportRouteStop.location.ilike(f"%{location.strip()}%"))
            .distinct()
        )
    return query.order_by(TransportRoute.shift, TransportRoute.bus_label, TransportRoute.id).all()
