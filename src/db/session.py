"""
Sessão e Engine do Banco de Dados Relacional

O `create_engine` do SQLAlchemy é preguiçoso — não abre conexão no import.
Nenhum módulo existente que não use `get_db()` é afetado pela presença
(ou ausência) do Postgres.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """Dependência FastAPI — fornece uma sessão por requisição e garante o fechamento."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
