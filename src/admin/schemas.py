"""Schemas Pydantic — Rotas Administrativas (/admin/*)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Cursos ───────────────────────────────────────────────────────────────────


class CourseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    name: str
    active: bool


class CourseCreateRequest(BaseModel):
    code: str
    name: str
    active: bool = True


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


# ── Corpo Docente ─────────────────────────────────────────────────────────────


class DisciplineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    name: str
    code: str | None
    period: int | None
    workload: int | None
    prerequisites_text: str | None


class DisciplineCreateRequest(BaseModel):
    course_id: int
    name: str
    code: str | None = None
    period: int | None = None
    workload: int | None = None
    prerequisites_text: str | None = None


class ProfessorDisciplineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    semester_year: str
    schedule_text: str | None
    room: str | None
    discipline: DisciplineOut


class ProfessorDisciplineAssignRequest(BaseModel):
    discipline_id: int
    semester_year: str
    schedule_text: str | None = None
    room: str | None = None


class ProfessorOut(BaseModel):
    """Representação de um Professor (ver src/db/models.py), com disciplinas aninhadas."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    email_secondary: str | None
    department: str | None
    course_id: int | None
    area: str | None
    lattes_url: str | None
    personal_site_url: str | None
    is_nde: bool
    nde_role: str | None
    bio: str | None
    created_at: datetime
    disciplines: list[ProfessorDisciplineOut] = Field(default_factory=list)


class ProfessorCreateRequest(BaseModel):
    name: str
    email: str
    email_secondary: str | None = None
    department: str | None = None
    course_id: int | None = None
    area: str | None = None
    lattes_url: str | None = None
    personal_site_url: str | None = None
    is_nde: bool = False
    nde_role: str | None = None
    bio: str | None = None


class ProfessorUpdateRequest(BaseModel):
    """Payload de PATCH /admin/professors/{id} — todos os campos opcionais."""

    name: str | None = None
    email: str | None = None
    email_secondary: str | None = None
    department: str | None = None
    course_id: int | None = None
    area: str | None = None
    lattes_url: str | None = None
    personal_site_url: str | None = None
    is_nde: bool | None = None
    nde_role: str | None = None
    bio: str | None = None


# ── Calendário Acadêmico ────────────────────────────────────────────────────


class AcademicEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int | None
    title: str
    start_date: date
    end_date: date | None
    category: str | None
    legal_reference: str | None
    campus: str | None
    academic_period: str | None


class AcademicEventCreateRequest(BaseModel):
    title: str
    start_date: date
    end_date: date | None = None
    course_id: int | None = None
    category: str | None = None
    legal_reference: str | None = None
    campus: str | None = None
    academic_period: str | None = None


class AcademicEventUpdateRequest(BaseModel):
    """Payload de PATCH /admin/academic-events/{id} — todos os campos opcionais."""

    title: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    course_id: int | None = None
    category: str | None = None
    legal_reference: str | None = None
    campus: str | None = None
    academic_period: str | None = None
