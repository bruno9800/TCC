"""
Modelos ORM — Camada de Dados Relacional

Entidades administrativas introduzidas na v2 (cursos, base de conhecimento,
documentos, professores). O pipeline RAG em si (chunks, embeddings) continua
vivendo no ChromaDB — aqui ficam apenas os dados relacionais que o painel
administrativo e o agente multi-tool precisam consultar com precisão exata.

Ver PLANO_V2.md, Seção 7.1, para a justificativa de cada tabela.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Cursos e Bases de Conhecimento ──────────────────────────────────────────────


class Course(Base):
    """Um curso de graduação (ex.: Engenharia de Computação)."""

    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class KnowledgeBase(Base):
    """
    Taxonomia de bases de conhecimento (ex.: 'regulamentos', 'manual_aluno', 'ppc').

    Não é uma collection separada no ChromaDB — é um metadado (`kb_slug`) usado
    para organização/filtragem opcional dentro de uma única collection.
    Ver PLANO_V2.md, Seção 6.1.
    """

    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None]
    chunking_strategy: Mapped[str] = mapped_column(String(20), default="legal")
    scope: Mapped[str] = mapped_column(String(20), default="institutional")


# ── Documentos ───────────────────────────────────────────────────────────────────


class Document(Base):
    """Registro administrativo de um documento normativo (PDF/DOCX)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    knowledge_base_id: Mapped[int] = mapped_column(ForeignKey("knowledge_bases.id"))
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id"), nullable=True)

    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(300))
    storage_path: Mapped[str] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(64))
    category: Mapped[str | None] = mapped_column(String(100))

    status: Mapped[str] = mapped_column(String(20), default="processing")
    version: Mapped[int] = mapped_column(default=1)
    revoked: Mapped[bool] = mapped_column(default=False)
    revoked_reason: Mapped[str | None]
    superseded_by_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id"), nullable=True
    )

    uploaded_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("admin_users.id"), nullable=True
    )
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    indexed_at: Mapped[datetime | None]

    knowledge_base: Mapped["KnowledgeBase"] = relationship()
    course: Mapped["Course | None"] = relationship()
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """
    Espelho leve dos vetores indexados no ChromaDB.

    Não duplica o conteúdo do chunk (isso vive no ChromaDB) — guarda apenas o
    `chroma_id` para permitir apagar vetores órfãos em uma reindexação
    (ver PLANO_V2.md, Seção 6.7 / diagnóstico D2). Populado a partir da Fase 1.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    chroma_id: Mapped[str] = mapped_column(String(300))
    article_id: Mapped[str | None] = mapped_column(String(50))
    hierarchy: Mapped[str | None] = mapped_column(String(500))
    token_count: Mapped[int | None]

    document: Mapped["Document"] = relationship(back_populates="chunks")


class IngestionJob(Base):
    """Rastreia o progresso do pipeline ETL→chunk→embed→index para um documento."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    stage: Mapped[str] = mapped_column(String(20))  # etl | chunking | embedding | indexing
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|done|failed
    error_message: Mapped[str | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


# ── Corpo Docente ────────────────────────────────────────────────────────────────


class Professor(Base):
    """
    Dado estruturado de professor — consultado via Tool (SQL), não via RAG.

    Ver PLANO_V2.md, Seção 5: fatos exatos (nome, e-mail, disciplinas) não
    devem depender de recall de embedding.
    """

    __tablename__ = "professors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    department: Mapped[str | None] = mapped_column(String(200))
    bio: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    disciplines: Mapped[list["ProfessorDiscipline"]] = relationship(back_populates="professor")


class Discipline(Base):
    """Disciplina de um curso (ex.: 'Cálculo I', período 1, Engenharia de Computação)."""

    __tablename__ = "disciplines"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    name: Mapped[str] = mapped_column(String(200))
    code: Mapped[str | None] = mapped_column(String(20))
    period: Mapped[int | None]
    workload: Mapped[int | None]

    course: Mapped["Course"] = relationship()
    professors: Mapped[list["ProfessorDiscipline"]] = relationship(back_populates="discipline")


class ProfessorDiscipline(Base):
    """
    Associação M2M professor↔disciplina, com atributos de oferta
    (semestre/ano, horário, sala) — um professor pode lecionar mais de uma
    disciplina, e uma disciplina pode ter mais de um professor ao longo dos
    semestres.
    """

    __tablename__ = "professor_disciplines"
    __table_args__ = (
        UniqueConstraint("professor_id", "discipline_id", "semester_year"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    professor_id: Mapped[int] = mapped_column(ForeignKey("professors.id"))
    discipline_id: Mapped[int] = mapped_column(ForeignKey("disciplines.id"))
    semester_year: Mapped[str] = mapped_column(String(10))  # ex: "2026.1"
    schedule_text: Mapped[str | None] = mapped_column(String(200))
    room: Mapped[str | None] = mapped_column(String(50))

    professor: Mapped["Professor"] = relationship(back_populates="disciplines")
    discipline: Mapped["Discipline"] = relationship(back_populates="professors")


# ── Administração ────────────────────────────────────────────────────────────────


class AdminUser(Base):
    """Usuário do futuro painel administrativo — separado da API key pública."""

    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="admin")  # admin | editor
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_login: Mapped[datetime | None]
