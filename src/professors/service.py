"""
ProfessorService — CRUD sobre Professor/Discipline/ProfessorDiscipline

Fecha a decisão D9 do EVOLUTION.md (v1): professores são dado estruturado,
consultados via SQL — não via RAG (zero tolerância a alucinação em fatos
exatos como e-mail/área). Ver PLANO_V2.md, Seção 5.

Discipline/ProfessorDiscipline permanecem como infraestrutura para quando
houver dado curricular preciso (Fase 5, importação do PPC) — não são
populados a partir de texto livre de "área principal".
"""

from __future__ import annotations

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from src.db.models import Discipline, Professor, ProfessorDiscipline


# ── Professores ────────────────────────────────────────────────────────────────


def create_professor(
    db: Session,
    name: str,
    email: str,
    email_secondary: str | None = None,
    department: str | None = None,
    course_id: int | None = None,
    area: str | None = None,
    lattes_url: str | None = None,
    personal_site_url: str | None = None,
    is_nde: bool = False,
    nde_role: str | None = None,
    bio: str | None = None,
) -> Professor:
    professor = Professor(
        name=name,
        email=email,
        email_secondary=email_secondary,
        department=department,
        course_id=course_id,
        area=area,
        lattes_url=lattes_url,
        personal_site_url=personal_site_url,
        is_nde=is_nde,
        nde_role=nde_role,
        bio=bio,
    )
    db.add(professor)
    db.commit()
    db.refresh(professor)
    return professor


def _with_disciplines(query):
    return query.options(
        joinedload(Professor.disciplines).joinedload(ProfessorDiscipline.discipline)
    )


def list_professors(
    db: Session,
    course_id: int | None = None,
    area: str | None = None,
    name: str | None = None,
) -> list[Professor]:
    query = _with_disciplines(db.query(Professor))
    if course_id is not None:
        # Inclui professores institucionais (course_id IS NULL), não só os do curso pedido.
        query = query.filter(
            or_(Professor.course_id == course_id, Professor.course_id.is_(None))
        )
    if area is not None:
        query = query.filter(Professor.area.ilike(f"%{area}%"))
    if name is not None:
        query = query.filter(Professor.name.ilike(f"%{name}%"))
    return query.order_by(Professor.name).all()


def get_professor(db: Session, professor_id: int) -> Professor | None:
    return _with_disciplines(db.query(Professor)).filter(Professor.id == professor_id).one_or_none()


def update_professor(db: Session, professor_id: int, **fields) -> Professor:
    """Atualiza os campos fornecidos (o router deve passar `exclude_unset=True`)."""
    professor = db.get(Professor, professor_id)
    if professor is None:
        raise ValueError(f"Professor {professor_id} não encontrado")

    for key, value in fields.items():
        setattr(professor, key, value)

    db.commit()
    db.refresh(professor)
    return professor


def delete_professor(db: Session, professor_id: int) -> None:
    professor = db.get(Professor, professor_id)
    if professor is None:
        raise ValueError(f"Professor {professor_id} não encontrado")
    db.delete(professor)  # cascade remove ProfessorDiscipline (relationship já configurada)
    db.commit()


# ── Disciplinas (infraestrutura para a Fase 5 — dado curricular do PPC) ────────


def create_discipline(
    db: Session,
    course_id: int,
    name: str,
    code: str | None = None,
    period: int | None = None,
    workload: int | None = None,
    prerequisites_text: str | None = None,
) -> Discipline:
    discipline = Discipline(
        course_id=course_id,
        name=name,
        code=code,
        period=period,
        workload=workload,
        prerequisites_text=prerequisites_text,
    )
    db.add(discipline)
    db.commit()
    db.refresh(discipline)
    return discipline


def list_disciplines(db: Session, course_id: int | None = None) -> list[Discipline]:
    query = db.query(Discipline)
    if course_id is not None:
        query = query.filter(Discipline.course_id == course_id)
    return query.order_by(Discipline.period, Discipline.name).all()


def assign_discipline(
    db: Session,
    professor_id: int,
    discipline_id: int,
    semester_year: str,
    schedule_text: str | None = None,
    room: str | None = None,
) -> ProfessorDiscipline:
    """
    Associa um professor a uma disciplina em um semestre. Idempotente: se a
    associação (professor, disciplina, semestre) já existir, atualiza
    horário/sala em vez de estourar IntegrityError na UniqueConstraint.
    """
    professor = db.get(Professor, professor_id)
    if professor is None:
        raise ValueError(f"Professor {professor_id} não encontrado")
    discipline = db.get(Discipline, discipline_id)
    if discipline is None:
        raise ValueError(f"Discipline {discipline_id} não encontrada")

    existing = (
        db.query(ProfessorDiscipline)
        .filter_by(professor_id=professor_id, discipline_id=discipline_id, semester_year=semester_year)
        .one_or_none()
    )
    if existing is not None:
        existing.schedule_text = schedule_text
        existing.room = room
        db.commit()
        db.refresh(existing)
        return existing

    assignment = ProfessorDiscipline(
        professor_id=professor_id,
        discipline_id=discipline_id,
        semester_year=semester_year,
        schedule_text=schedule_text,
        room=room,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment
