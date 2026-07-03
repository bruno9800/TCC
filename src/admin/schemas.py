"""Schemas Pydantic — Rotas Administrativas (/admin/*)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentOut(BaseModel):
    """Representação administrativa de um Document (ver src/db/models.py)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    course_id: int | None
    title: str
    filename: str
    category: str | None
    status: str
    version: int
    revoked: bool
    revoked_reason: str | None
    superseded_by_document_id: int | None
    uploaded_at: datetime
    indexed_at: datetime | None


class DocumentUpdateRequest(BaseModel):
    """Payload de PATCH /admin/documents/{id} — todos os campos opcionais."""

    title: str | None = None
    category: str | None = None
    course_id: int | None = None
    knowledge_base_id: int | None = None
    revoked: bool | None = None
    revoked_reason: str | None = None
    superseded_by_document_id: int | None = None
