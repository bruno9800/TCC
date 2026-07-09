"""
Transport Router — Consulta Pública do Itinerário do Transporte Estudantil

Único endpoint público (protegido pela x-api-key, mesmo padrão de
/academic-events e /professors). Os dados são populados exclusivamente pela
importação estruturada (POST /admin/imports, import_type='transport').
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.admin.schemas import TransportRouteOut
from src.db.session import get_db
from src.transport import service as transport_service

router = APIRouter()


@router.get(
    "",
    response_model=list[TransportRouteOut],
    summary="Lista rotas do transporte estudantil (com paradas, em ordem de itinerário)",
)
async def list_routes(
    semester: str | None = None,
    shift: str | None = None,
    location: str | None = None,
    db: Session = Depends(get_db),
) -> list[TransportRouteOut]:
    return transport_service.list_routes(
        db, semester=semester, shift=shift, location=location
    )
