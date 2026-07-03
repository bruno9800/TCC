#!/usr/bin/env python3
"""
Script de seed — Corpo Docente do Colegiado de Engenharia da Computação (CECOMP/UNIVASF).

Dado real fornecido pelo usuário (não é dado de teste/descartável): 7 membros
do Núcleo Docente Estruturante (NDE) — um deles Coordenador — e 8 demais
professores do colegiado. Vinculado ao curso ENGCOMP semeado na Fase 0.

Idempotente por e-mail (mesmo padrão de seed_db.py/backfill_documents.py).

Uso:
    python scripts/seed_professors_engcomp.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Course, Professor
from src.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEPARTMENT = "Colegiado de Engenharia da Computação (CECOMP)"

# ── Núcleo Docente Estruturante (NDE) ────────────────────────────────────────
NDE_PROFESSORS = [
    dict(
        name="Brauliro Gonçalves Leal",
        email="brauliro.leal@univasf.edu.br",
        email_secondary="brauliro@pesquisador.cnpq.br",
        degree="Doutor",
        area="Modelagem Matemática e Computacional",
        lattes_url="http://lattes.cnpq.br/8388825387593034",
        personal_site_url="http://www.univasf.edu.br/~brauliro.leal",
        nde_role="Coordenador",
    ),
    dict(
        name="Jorge Luis Cavalcanti Ramos",
        email="jorge.cavalcanti@univasf.edu.br",
        degree="Doutor",
        area="Computação Gráfica, Matemática Discreta",
        lattes_url="http://lattes.cnpq.br/1438322656914569",
        personal_site_url="http://www.univasf.edu.br/~jorge.cavalcanti",
    ),
    dict(
        name="Jadsonlee da Silva Sá",
        email="jadsonlee.sa@univasf.edu.br",
        degree="Doutor",
        area="Sistemas Embarcados",
        lattes_url="http://lattes.cnpq.br/2010145273028144",
        personal_site_url="http://www.univasf.edu.br/~jadsonlee.sa",
    ),
    dict(
        name="Jairson Barbosa Rodrigues",
        email="jairson.rodrigues@univasf.edu.br",
        degree="Doutor",
        area="Sistemas Operacionais, Sistemas Distribuídos, Criptografia",
        lattes_url="http://lattes.cnpq.br/0036738410783279",
        personal_site_url="http://www.univasf.edu.br/~jairson.rodrigues",
    ),
    dict(
        name="Juracy Emanuel Magalhães da Franca",
        email="juracy.emanuel@univasf.edu.br",
        degree="Mestre",
        area="Automação, Robótica",
        lattes_url="http://lattes.cnpq.br/4900473312230462",
        personal_site_url="http://www.univasf.edu.br/~juracy.emanuel",
    ),
    dict(
        name="Marcus Vinícius Midena Ramos",
        email="marcus.ramos@univasf.edu.br",
        degree="Doutor",
        area="Teoria da Computação",
        lattes_url="http://lattes.cnpq.br/7833733286842741",
        personal_site_url="http://www.univasf.edu.br/~marcus.ramos",
    ),
    dict(
        name="Rafael Moura Duarte",
        email="rafael.mouraduarte@univasf.edu.br",
        degree="Doutor",
        area="Processamento de Sinais, Sistemas Eletrônicos e Inteligência Artificial",
        lattes_url="https://lattes.cnpq.br/9518314608063013",
        personal_site_url=None,
    ),
]

# ── Demais Professores ────────────────────────────────────────────────────────
OTHER_PROFESSORS = [
    dict(
        name="Ana Emília de Melo Queiroz",
        email="ana.queiroz@univasf.edu.br",
        degree="Doutora",
        area="Algoritmos, Estruturas de Dados",
        lattes_url="http://lattes.cnpq.br/2923710960629772",
        personal_site_url="http://www.univasf.edu.br/~ana.queiroz",
    ),
    dict(
        name="Ana Júlia Fernandes de Oliveira Barros",
        email="anajulia.oliveira@univasf.edu.br",
        degree="Doutora",
        area="Telecomunicações",
        lattes_url=None,  # não disponível na fonte
        personal_site_url="http://www.univasf.edu.br/~anajulia.oliveira",
    ),
    dict(
        name="Fábio Nelson de Sousa Pereira",
        email="fabio.nelson@univasf.edu.br",
        degree="Mestre",
        area="Sistemas Embarcados, Redes de Sensores sem Fio",
        lattes_url="http://lattes.cnpq.br/3855845313174474",
        personal_site_url="http://www.univasf.edu.br/~fabio.nelson",
    ),
    dict(
        name="Marcelo Santos Linder",
        email="marcelo.linder@univasf.edu.br",
        degree="Mestre",
        area="Linguagens e Programação e Algoritmos",
        lattes_url="http://lattes.cnpq.br/0118309756941390",
        personal_site_url="http://www.univasf.edu.br/~marcelo.linder",
    ),
    dict(
        name="Mario Godoy Neto",
        email="mario.godoy@univasf.edu.br",
        degree="Doutor",
        area="Bancos de Dados",
        lattes_url="http://lattes.cnpq.br/6381727641321786",
        personal_site_url="http://www.univasf.edu.br/~mario.godoy",
    ),
    dict(
        name="Max Santana Rolemberg Farias",
        email="max.santana@univasf.edu.br",
        degree="Doutor",
        area="Redes de Computadores",
        lattes_url="http://lattes.cnpq.br/9688352609644792",
        personal_site_url="http://www.univasf.edu.br/~max.santana",
    ),
    dict(
        name="Ricardo Argenton Ramos",
        email="ricardo.aramos@univasf.edu.br",
        degree="Doutor",
        area="Engenharia de Software",
        lattes_url="http://lattes.cnpq.br/6190953685221120",
        personal_site_url="http://www.univasf.edu.br/~ricardo.aramos",
    ),
    dict(
        name="Rosalvo Ferreira de Oliveira Neto",
        email="rosalvo.oliveira@univasf.edu.br",
        degree="Doutor",
        area="Inteligência Artificial, Mineração de Dados",
        lattes_url="http://lattes.cnpq.br/9548186939653024",
        personal_site_url="http://www.univasf.edu.br/~rosalvo.oliveira",
    ),
]


def seed_professor(session, data: dict, course_id: int, is_nde: bool) -> None:
    existing = session.query(Professor).filter_by(email=data["email"]).one_or_none()
    if existing:
        # Já existe (criado antes do campo `degree` existir, Fase 3) — só
        # sincroniza o novo campo, sem recriar nem sobrescrever o resto.
        if data.get("degree") and existing.degree != data["degree"]:
            existing.degree = data["degree"]
            logger.info(f"  Atualizado (degree): {data['name']} → {data['degree']}")
        else:
            logger.info(f"  Já existe: {data['name']} ({data['email']})")
        return

    professor = Professor(
        name=data["name"],
        email=data["email"],
        email_secondary=data.get("email_secondary"),
        department=DEPARTMENT,
        course_id=course_id,
        area=data.get("area"),
        lattes_url=data.get("lattes_url"),
        personal_site_url=data.get("personal_site_url"),
        degree=data.get("degree"),
        is_nde=is_nde,
        nde_role=data.get("nde_role"),
    )
    session.add(professor)

    tag = ""
    if is_nde:
        tag = f" [NDE — {data['nde_role']}]" if data.get("nde_role") else " [NDE]"
    logger.info(f"  + {data['name']}{tag}")


def main():
    logger.info("=" * 60)
    logger.info("SEED — Corpo Docente CECOMP (Engenharia de Computação)")
    logger.info("=" * 60)

    session = SessionLocal()
    try:
        course = session.query(Course).filter_by(code="ENGCOMP").one_or_none()
        if course is None:
            logger.error(
                "Curso 'ENGCOMP' não encontrado. Rode `python scripts/seed_db.py` primeiro."
            )
            return

        for data in NDE_PROFESSORS:
            seed_professor(session, data, course_id=course.id, is_nde=True)
        for data in OTHER_PROFESSORS:
            seed_professor(session, data, course_id=course.id, is_nde=False)

        session.commit()
        total = len(NDE_PROFESSORS) + len(OTHER_PROFESSORS)
        logger.info(f"\nSeed concluído — {total} professores no total (7 NDE + 8 demais).")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
