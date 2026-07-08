#!/usr/bin/env python3
"""
Script de seed do banco relacional.

Cria os dados mínimos necessários para o sistema funcionar com a dimensão de
curso/base de conhecimento: o curso "Engenharia de Computação" e a knowledge
base "regulamentos" (única necessária hoje — todo o corpus atual pertence a
ela). Idempotente: pode ser executado múltiplas vezes sem duplicar registros.

Uso:
    python scripts/seed_db.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Course, KnowledgeBase
from src.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def seed_course(session) -> Course:
    course = session.query(Course).filter_by(code="ENGCOMP").one_or_none()
    if course:
        logger.info(f"  Curso já existe: {course.code} — {course.name}")
        return course

    course = Course(code="ENGCOMP", name="Engenharia de Computação", active=True)
    session.add(course)
    session.flush()
    logger.info(f"  Curso criado: {course.code} — {course.name}")
    return course


def seed_knowledge_base(session) -> KnowledgeBase:
    kb = session.query(KnowledgeBase).filter_by(slug="regulamentos").one_or_none()
    if kb:
        logger.info(f"  Knowledge base já existe: {kb.slug}")
        return kb

    kb = KnowledgeBase(
        slug="regulamentos",
        name="Regulamentos Institucionais",
        description=(
            "Estatuto, Regimento Geral e Resoluções da UNIVASF (PROEN/PROEX/PRPPGI) — "
            "corpus normativo já indexado no ChromaDB desde a v1."
        ),
        chunking_strategy="legal",
        scope="institutional",
    )
    session.add(kb)
    session.flush()
    logger.info(f"  Knowledge base criada: {kb.slug}")
    return kb


def main():
    logger.info("=" * 60)
    logger.info("SEED — Curso e Knowledge Base padrão")
    logger.info("=" * 60)

    session = SessionLocal()
    try:
        seed_course(session)
        seed_knowledge_base(session)
        session.commit()
        logger.info("Seed concluído com sucesso.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
