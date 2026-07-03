#!/usr/bin/env python3
"""
Script de seed — Matriz Curricular do curso de Engenharia de Computação (CECOMP/UNIVASF).

Dado real extraído do PPC (Projeto Pedagógico do Curso), Seção 4.2 (Matriz
Curricular, 1º a 10º período) e 4.2.12-4.2.14 (Disciplinas Optativas). Não é
dado de teste — fica no sistema, mesma filosofia de
scripts/seed_professors_engcomp.py.

Pré-requisito/co-requisito viram um campo de texto simples
(`prerequisites_text`) — não uma modelagem relacional de grafo, que seria
overengineering para o que o agente precisa responder.

Idempotente por (course_id, code) — só cria se não existir.

Uso:
    python scripts/seed_disciplines_engcomp.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Course, Discipline
from src.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cada entrada: (nome, sigla, período, carga_horária, pré-requisito/co-requisito em texto livre)
DISCIPLINES = [
    # ── 1º período ──────────────────────────────────────────────────────────
    ("Algoritmos e Programação para Computação", "APC", 1, 90, None),
    ("Cálculo Diferencial e Integral I", "CDI-I", 1, 60, None),
    ("Comunicação e Expressão", "CEXP", 1, 30, None),
    ("Geometria Analítica", "GA", 1, 60, None),
    ("Introdução à Engenharia de Computação", "IEC", 1, 30, None),
    ("Matemática Discreta", "MD", 1, 60, None),
    # ── 2º período ──────────────────────────────────────────────────────────
    ("Algoritmo e Estruturas de Dados", "AED", 2, 60, "Pré-requisito: APC"),
    ("Álgebra Linear", "AL", 2, 60, "Pré-requisito: GA"),
    ("Cálculo Diferencial e Integral II", "CDI-II", 2, 60, "Pré-requisito: CDI-I"),
    ("Eletrônica Digital", "ED", 2, 60, "Pré-requisito: IEC"),
    ("Laboratório de Eletrônica Digital", "LED", 2, 30, "Co-requisito: ED"),
    ("Física Teórica I", "FT-I", 2, 60, None),
    # ── 3º período ──────────────────────────────────────────────────────────
    ("Circuitos Elétricos", "CE", 3, 60, None),
    ("Laboratório de Circuitos Elétricos", "LCE", 3, 30, "Co-requisito: CE"),
    ("Lógica para Computação", "LC", 3, 60, "Pré-requisito: MD"),
    ("Física Experimental A", "FE-A", 3, 60, "Pré-requisito: FT-I; Co-requisito: FT-II"),
    ("Física Teórica II", "FT-II", 3, 60, "Pré-requisito: FT-I"),
    ("Organização e Arquitetura de Computadores", "OAC", 3, 60, "Pré-requisito: ED"),
    (
        "Laboratório de Organização e Arquitetura de Computadores",
        "LOAC",
        3,
        30,
        "Pré-requisito: LED; Co-requisito: OAC",
    ),
    # ── 4º período ──────────────────────────────────────────────────────────
    ("Cálculo Diferencial e Integral IV", "CDI-IV", 4, 60, "Pré-requisito: CDI-II"),
    ("Cálculo Numérico", "CN", 4, 60, "Pré-requisito: AL, CDI-II e APC"),
    ("Estatística e Probabilidade para Computação", "EPC", 4, 60, "Pré-requisito: CDI-I"),
    ("Eletrônica Analógica", "EA", 4, 60, "Pré-requisito: CE"),
    ("Laboratório de Eletrônica Analógica", "LEA", 4, 30, "Pré-requisito: LCE; Co-requisito: EA"),
    ("Física Teórica III", "FT-III", 4, 60, "Pré-requisito: FT-II e CDI-II"),
    ("Programação Orientada a Objetos", "POO", 4, 60, "Pré-requisito: APC"),
    # ── 5º período ──────────────────────────────────────────────────────────
    ("Análise de Sinais e Sistemas", "ASS", 5, 60, "Pré-requisito: CDI-IV"),
    ("Banco de Dados I", "BD-I", 5, 60, "Co-requisito: ES-I"),
    ("Engenharia de Software I", "ES-I", 5, 60, "Pré-requisito: POO; Co-requisito: BD-I"),
    ("Rede de Computadores", "RC", 5, 60, "Pré-requisito: OAC"),
    ("Laboratório de Rede de Computadores", "LRC", 5, 30, "Co-requisito: RC"),
    ("Sistemas Microcontrolados", "SM", 5, 60, "Pré-requisito: APC, EA, LEA e OAC"),
    ("Sistemas Operacionais", "SO", 5, 60, "Pré-requisito: OAC e AED"),
    # ── 6º período ──────────────────────────────────────────────────────────
    ("Administração para Engenharia", "AE", 6, 30, None),
    ("Banco de Dados II", "BD-II", 6, 30, "Pré-requisito: BD-I"),
    ("Engenharia de Software II", "ES-II", 6, 60, "Pré-requisito: ES-I e BD-I"),
    ("Instrumentação Eletrônica", "IE", 6, 60, "Pré-requisito: SM"),
    ("Inteligência Artificial", "IA", 6, 60, "Pré-requisito: AED e LC"),
    ("Linguagens Formais e Autômatos", "LFA", 6, 60, "Pré-requisito: APC e MD"),
    ("Sistemas Distribuídos I", "SD-I", 6, 60, "Pré-requisito: SO, RC"),
    # ── 7º período ──────────────────────────────────────────────────────────
    ("Compiladores", "CO", 7, 60, "Pré-requisito: AED, LFA e OAC"),
    ("Computação Gráfica", "CG", 7, 60, "Pré-requisito: AED e GA"),
    ("Introdução a Engenharia Econômica", "IEE", 7, 30, None),
    ("Sistemas de Controle I", "SC-I", 7, 60, "Pré-requisito: ASS"),
    ("Meio Ambiente e Desenv. Sustentável", "MADS", 7, 30, None),
    ("Princípios de Telecomunicações", "PT", 7, 60, "Pré-requisito: ASS"),
    ("Metodologia da Pesquisa", "MP", 7, 30, "Pré-requisito: CEXP"),
    # ── 8º período ──────────────────────────────────────────────────────────
    ("Aspectos Legais para Computação", "ALC", 8, 30, None),
    ("Modelagem e Simulação Discreta", "MSD", 8, 60, "Pré-requisito: EPC, OAC, CN e ES-I"),
    ("Sistemas de Controle II", "SC-II", 8, 60, "Pré-requisito: SC-I"),
    ("Teoria da Computação", "TC", 8, 60, "Pré-requisito: LFA"),
    ("Trabalho de Conclusão de Curso I", "TCC-I", 8, 60, "Pré-requisito: 70% do curso"),
    ("Eletiva I", "ELT-I", 8, 60, None),
    ("Optativa I", "OPT-I", 8, 60, None),
    # ── 9º período ──────────────────────────────────────────────────────────
    ("Avaliação de Desempenho de Sistemas", "ADS", 9, 60, "Pré-requisito: MSD, RC e SD-I"),
    ("Empreendedorismo", "EMP", 9, 30, None),
    ("Núcleo Temático", "NT", 9, 120, None),
    ("Optativa II", "OPT-II", 9, 60, None),
    ("Optativa III", "OPT-III", 9, 60, None),
    ("Trabalho de Conclusão de Curso II", "TCC-II", 9, 60, "Pré-requisito: TCC-I"),
    # ── 10º período ─────────────────────────────────────────────────────────
    ("Estágio Supervisionado", "ES", 10, 240, "Pré-requisito: 50% do curso"),
    ("Eletiva II", "ELT-II", 10, 60, None),
    ("Optativa IV", "OPT-IV", 10, 60, None),
    # ── Optativas: Sistemas Computacionais ─────────────────────────────────
    ("Sistemas Embarcados", "SE", None, 60, "Pré-requisito: SM e SO"),
    ("Sistemas de Tempo Real", "STR", None, 60, "Pré-requisito: SM e SO"),
    ("Elementos de Criptografia e Segurança", "ECS", None, 60, "Pré-requisito: APC; Co-requisito: RC"),
    ("Sistemas Distribuídos II", "SD-II", None, 60, "Pré-requisito: SD-I"),
    ("Projeto de Sistemas Operacionais", "PSO", None, 60, "Pré-requisito: SO"),
    ("Tópicos Avançados em Sistemas Computacionais", "TASC", None, 60, None),
    # ── Optativas: Controle, Automação e Robótica ──────────────────────────
    ("Automação Industrial", "AI", None, 60, "Pré-requisito: SC-I e IE"),
    ("Robótica I", "R-I", None, 60, "Pré-requisito: EPC, POO e CN"),
    ("Robótica II", "R-II", None, 60, "Pré-requisito: SC-I"),
    ("Aprendizado Profundo", "AP", None, 60, "Pré-requisito: IA"),
    ("Tópicos Avançados em Automação", "TAA", None, 60, None),
    ("Língua Brasileira de Sinais", "LIBRAS", None, 60, "Pré-requisito: CEXP"),
    # ── Optativas: Engenharia de Software ──────────────────────────────────
    ("Engenharia de Software III", "ES-III", None, 60, "Pré-requisito: ES-I e BD-I"),
    ("Verificação, Validação e Teste de Software", "VVTS", None, 60, "Pré-requisito: ES-I"),
    ("Fábrica de Software", "FS", None, 60, "Pré-requisito: ES-II"),
    ("Banco de Dados Avançados", "BDA", None, 60, "Pré-requisito: BD-II e ES-II"),
    ("Programação Avançada", "PA", None, 60, "Pré-requisito: SO"),
    ("Tópicos Avançados em Engenharia de Software", "TAES", None, 60, None),
]


def main():
    logger.info("=" * 60)
    logger.info("SEED — Matriz Curricular (Engenharia de Computação)")
    logger.info("=" * 60)

    session = SessionLocal()
    try:
        course = session.query(Course).filter_by(code="ENGCOMP").one_or_none()
        if course is None:
            logger.error(
                "Curso 'ENGCOMP' não encontrado. Rode `python scripts/seed_db.py` primeiro."
            )
            return

        created, skipped = 0, 0
        for name, code, period, workload, prereq in DISCIPLINES:
            existing = (
                session.query(Discipline)
                .filter_by(course_id=course.id, code=code)
                .one_or_none()
            )
            if existing:
                skipped += 1
                continue

            discipline = Discipline(
                course_id=course.id,
                name=name,
                code=code,
                period=period,
                workload=workload,
                prerequisites_text=prereq,
            )
            session.add(discipline)
            created += 1
            logger.info(f"  + {code} — {name} (período {period or 'optativa'})")

        session.commit()
        logger.info(f"\nSeed concluído — {created} criadas, {skipped} já existiam.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
