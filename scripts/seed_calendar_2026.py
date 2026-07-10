#!/usr/bin/env python3
"""
Script de seed — Calendário Acadêmico de Graduação 2026 (UNIVASF).

Dado real extraído do PDF "Calendário Acadêmico Graduação 2026" (janeiro/2026
a março/2027): início/fim de período letivo, prazos de matrícula/rematrícula/
trancamento, feriados nacionais/estaduais/municipais por campus, colação de
grau, exames finais e demais prazos administrativos. Não é dado de teste —
fica no sistema, mesma filosofia de scripts/seed_professors_engcomp.py e
scripts/seed_disciplines_engcomp.py.

`course_id=None` em todos os eventos: o calendário acadêmico vale para todos
os cursos da UNIVASF, não é específico de Engenharia de Computação.

Idempotente por (title, start_date) — só cria se não existir.

Uso:
    python scripts/seed_calendar_2026.py
"""

import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import AcademicEvent
from src.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _d(iso: str | None) -> date | None:
    return date.fromisoformat(iso) if iso else None


# Cada entrada: (título, data_início, data_fim, categoria, referência_legal, campus, período_letivo)
EVENTS: list[tuple[str, str, str | None, str, str | None, str | None, str | None]] = [
    # ===== Recesso de virada de ano 2025/2026 =====
    ("Recesso acadêmico e administrativo - Feriado Nacional - Confraternização Universal",
     "2025-12-31", "2026-01-02", "feriado", None, None, None),
    ("Prazo para os discentes solicitarem reintegração ao curso para o Período 2026.1",
     "2026-01-15", "2026-01-22", "matrícula", "Art. 63 da Resolução nº 08/2015 - Conuni", None, "2026.1"),
    ("Feriado Municipal para o campus Senhor do Bonfim - Padroeiro de Senhor do Bonfim/BA",
     "2026-01-17", None, "feriado", None, "SBF", None),
    ("Prazo para abertura de processo de colação de grau em gabinete para os discentes que não "
     "participarão das solenidades regulares do Período 2025.2",
     "2026-01-19", "2026-01-30", "colação", "Resolução nº 18/2017 - Conuni", None, "2025.2"),
    ("Último dia para os Colegiados divulgarem nas páginas dos cursos os programas dos componentes "
     "curriculares, duração, requisitos, qualificação dos professores, recursos disponíveis e "
     "critérios de avaliação para o Período 2026.1",
     "2026-01-30", None, "planejamento",
     "§ 1º do Art. 47 da Lei nº 9.394/1996, com redação dada pela Lei nº 13.168/2015", None, "2026.1"),

    # ===== Fevereiro 2026 =====
    ("Colação de Grau com solenidade dos formandos 2025.2",
     "2026-02-02", "2026-02-27", "colação", "Resolução nº 18/2017 - Conuni", None, "2025.2"),
    ("Matrícula presencial dos estudantes do Programa PEC-G para o Período 2026.1 "
     "(entrega de documentação na SRCA)",
     "2026-02-10", "2026-02-11", "matrícula", None, None, "2026.1"),
    ("Recesso acadêmico e administrativo - Ponto Facultativo - Carnaval/Quarta-feira de Cinzas",
     "2026-02-16", "2026-02-18", "feriado", None, None, None),
    ("Preparação para a matrícula do Período 2026.1 (SIG@ indisponível)",
     "2026-02-19", None, "matrícula", None, None, "2026.1"),
    ("Matrícula dos alunos veteranos para o Período 2026.1",
     "2026-02-20", "2026-02-23", "matrícula", None, None, "2026.1"),
    ("Data limite para as Coordenações de Curso encaminharem à SRCA os requerimentos de matrícula "
     "com expansão de carga horária para o Período 2026.1",
     "2026-02-23", None, "matrícula", None, None, "2026.1"),
    ("Registro de matrícula no SIG@ pela SRCA: disciplinas com alteração (quebra) de pré-requisito; "
     "aluno especial (disciplinas isoladas); mobilidade acadêmica (Convênios Andifes e InterIES); "
     "Programa PEC-G",
     "2026-02-24", "2026-02-25", "matrícula", None, None, "2026.1"),
    ("Ajuste da oferta pelas Coordenações de Curso",
     "2026-02-26", "2026-02-27", "matrícula", None, None, "2026.1"),
    ("Resultado das matrículas realizadas pelos veteranos, após as 22 horas",
     "2026-02-27", None, "matrícula", None, None, "2026.1"),
    ("Matrícula de retardatários e Modificação/correção de matrícula pelos alunos para o "
     "Período 2026.1",
     "2026-02-28", "2026-03-03", "matrícula", None, None, "2026.1"),
    ("Reabertura da caderneta eletrônica referente ao Período 2025.2",
     "2026-02-28", "2026-03-14", "outro", None, None, "2025.2"),

    # ===== Março 2026 =====
    ("Início das aulas do Período Letivo 2026.1",
     "2026-03-02", None, "período_letivo", None, None, "2026.1"),
    ("Semana de acolhimento e integração dos discentes",
     "2026-03-02", "2026-03-06", "período_letivo", None, None, "2026.1"),
    ("Registro de matrícula pela SRCA: somente disciplinas com expansão de carga horária, conforme "
     "requerimentos aprovados e encaminhados pelas Coordenações de Curso",
     "2026-03-04", "2026-03-05", "matrícula", None, None, "2026.1"),
    ("Feriado Estadual para os campi no Estado de Pernambuco - Data Magna de Pernambuco",
     "2026-03-06", None, "feriado", None, None, None),
    ("Ajuste da oferta pelas Coordenações de Curso (após matrícula de retardatários e modificação "
     "de matrícula)",
     "2026-03-06", "2026-03-09", "matrícula", None, None, "2026.1"),
    ("Resultado das matrículas de retardatários e modificação de matrículas realizadas pelos "
     "discentes, após as 22 horas",
     "2026-03-09", None, "matrícula", None, None, "2026.1"),
    ("Colação de grau em gabinete dos formandos 2025.2 que não participaram das solenidades "
     "regulares",
     "2026-03-09", "2026-03-20", "colação", None, None, "2025.2"),
    ("Matrículas em disciplinas com vagas ociosas pelos alunos",
     "2026-03-10", "2026-03-11", "matrícula", None, None, "2026.1"),
    ("Ajuste da oferta pelas Coordenações de Curso (após a matrícula em vagas ociosas)",
     "2026-03-12", None, "matrícula", None, None, "2026.1"),
    ("Resultado das matrículas em disciplinas com vagas ociosas, após as 22 horas",
     "2026-03-12", None, "matrícula", None, None, "2026.1"),
    ("Prazo para os discentes solicitarem matrícula extemporânea para o Período 2026.1",
     "2026-03-12", "2026-03-18", "matrícula", "Art. 51, § 2º da Resolução nº 08/2015 - Conuni",
     None, "2026.1"),
    ("Data limite para as Coordenações de Curso realizarem as matrículas extemporâneas no SIG@ "
     "para o Período 2026.1",
     "2026-03-27", None, "matrícula", "Art. 51, § 3º da Resolução nº 08/2015 - Conuni", None, "2026.1"),

    # ===== Abril 2026 =====
    ("Recesso acadêmico e administrativo - Feriado Nacional - Paixão de Cristo/Semana Santa",
     "2026-04-02", "2026-04-04", "feriado", None, None, None),
    ("Trancamento do período ou cancelamento de disciplinas (com ônus), referente a 2026.1",
     "2026-04-06", "2026-04-10", "trancamento", None, None, "2026.1"),
    ("Prazo para os discentes solicitarem dispensa de disciplinas, carga horária eletiva e "
     "optativa para o Período 2026.2",
     "2026-04-13", "2026-04-29", "matrícula", None, None, "2026.2"),
    ("Recesso acadêmico e administrativo - Feriado Nacional - Tiradentes",
     "2026-04-20", "2026-04-21", "feriado", None, None, None),
    ("Feriado Municipal para o campus Salgueiro - Emancipação Política de Salgueiro/PE",
     "2026-04-30", None, "feriado", None, "SAL", None),

    # ===== Maio 2026 =====
    ("Recesso acadêmico e administrativo - Feriado Nacional - Dia Mundial do Trabalho",
     "2026-05-01", "2026-05-02", "feriado", None, None, None),
    ("Prazo para abertura de processo de atividades complementares para registro no Período 2026.1",
     "2026-05-04", "2026-05-15", "outro", None, None, "2026.1"),
    ("Prazo para abertura de processos de alteração (quebra) de pré-requisito para o Período "
     "2026.2 pelo aluno ou pelos NDEs/Coordenações dos cursos de graduação",
     "2026-05-18", "2026-05-29", "outro", None, None, "2026.2"),
    ("Feriado Municipal para o campus Senhor do Bonfim - Aniversário da cidade de Senhor do "
     "Bonfim/BA",
     "2026-05-28", None, "feriado", None, "SBF", None),

    # ===== Junho 2026 =====
    ("Planejamento da oferta de componentes curriculares para o Período 2026.2 pelos Colegiados "
     "Acadêmicos (19/06 é a data limite para divulgação do planejamento)",
     "2026-06-01", "2026-06-12", "planejamento", None, None, "2026.2"),
    ("Recesso acadêmico e administrativo - Ponto facultativo - Corpus Christi",
     "2026-06-04", "2026-06-06", "feriado", None, None, None),
    ("Recesso acadêmico (Decisão nº 7/2026 - Conuni, alteração do Anexo I da Resolução nº "
     "13/2025 - Conuni)",
     "2026-06-05", "2026-06-06", "feriado", "Decisão nº 7/2026 - Conuni", None, None),
    ("Prazo para solicitação de matrícula especial/disciplinas isoladas (estudantes de outras "
     "IES) para o Período 2026.2",
     "2026-06-08", "2026-06-12", "matrícula", None, None, "2026.2"),
    ("Prazo para as Coordenações de Curso instruírem os processos de alteração (quebra) de "
     "pré-requisitos para o Período 2026.2 (parecer do docente, parecer do NDE, aprovação do "
     "Colegiado e envio à SRCA)",
     "2026-06-08", "2026-06-26", "outro", None, None, "2026.2"),
    ("Feriado Municipal para o campus Salgueiro - Padroeiro de Salgueiro/PE",
     "2026-06-13", None, "feriado", None, "SAL", None),
    ("Data limite para as Coordenações de Curso enviarem os processos de atividades "
     "complementares para registro no Período 2026.1",
     "2026-06-19", None, "outro", None, None, "2026.1"),
    ("Recesso acadêmico e administrativo - Feriado Regional - São João",
     "2026-06-23", "2026-06-24", "feriado", None, None, None),
    ("Feriado Municipal para o campus Serra da Capivara - Aniversário da cidade de São Raimundo "
     "Nonato/PI",
     "2026-06-25", None, "feriado", None, "SRN", None),
    ("Oferta de componentes curriculares para o Período 2026.2 pelas Coordenações de Curso "
     "no SIG@",
     "2026-06-29", "2026-07-10", "planejamento", None, None, "2026.2"),

    # ===== Julho 2026 =====
    ("Feriado Estadual - Emancipação da Bahia (somente para os campi no Estado da Bahia)",
     "2026-07-02", None, "feriado", None, None, None),
    ("Prazo para os discentes solicitarem reintegração ao curso para o Período 2026.2",
     "2026-07-06", "2026-07-15", "matrícula", "Art. 63 da Resolução nº 08/2015 - Conuni", None, "2026.2"),
    ("Data limite para as Coordenações de Curso encaminharem à SRCA os processos de dispensa de "
     "disciplinas, alteração (quebra) de pré-requisitos e matrícula especial/disciplinas isoladas "
     "(estudantes de outras IES) para o Período 2026.2",
     "2026-07-10", None, "outro", None, None, "2026.2"),
    ("Último dia para os Colegiados divulgarem nas páginas dos cursos os programas dos componentes "
     "curriculares, duração, requisitos, qualificação dos professores, recursos disponíveis e "
     "critérios de avaliação para o Período 2026.2",
     "2026-07-10", None, "planejamento",
     "§ 1º do Art. 47 da Lei nº 9.394/1996, com redação dada pela Lei nº 13.168/2015", None, "2026.2"),
    ("Feriado Municipal para o campus Juazeiro - Aniversário da cidade de Juazeiro/BA",
     "2026-07-15", None, "feriado", None, "JUA", None),
    ("Último dia de aulas do Período 2026.1",
     "2026-07-18", None, "período_letivo", None, None, "2026.1"),
    ("Início do prazo para preenchimento dos planos de ensino na caderneta eletrônica para o "
     "Período 2026.2",
     "2026-07-20", None, "outro", None, None, "2026.2"),
    ("Exames Finais do Período 2026.1",
     "2026-07-20", "2026-07-24", "exames", None, None, "2026.1"),
    ("Prazo para os discentes solicitarem matrícula com expansão de carga horária para o "
     "Período 2026.2 às Coordenações de Curso",
     "2026-07-20", "2026-07-24", "matrícula", None, None, "2026.2"),
    ("Matrícula presencial dos estudantes do Programa PEC-G para o Período 2026.2 "
     "(entrega de documentação na SRCA)",
     "2026-07-21", "2026-07-22", "matrícula", None, None, "2026.2"),
    ("Feriado Municipal para o campus Paulo Afonso - Aniversário da cidade de Paulo Afonso/BA",
     "2026-07-28", None, "feriado", None, "PAV", None),
    ("Último dia para lançamento das notas e faltas do Período 2026.1 pelos docentes. Data limite "
     "para finalizar preenchimento da caderneta eletrônica referente a 2026.1",
     "2026-07-29", None, "outro", "Resolução nº 16/2021 - Conuni", None, "2026.1"),
    ("Encerramento do Período Letivo 2026.1",
     "2026-07-29", None, "período_letivo", None, None, "2026.1"),
    ("Preparação para a matrícula do Período 2026.2 (SIG@ indisponível)",
     "2026-07-30", None, "matrícula", None, None, "2026.2"),
    ("Matrícula dos alunos veteranos para o Período 2026.2",
     "2026-07-31", "2026-08-03", "matrícula", None, None, "2026.2"),

    # ===== Agosto 2026 =====
    ("Data limite para as Coordenações de Curso encaminharem à SRCA os requerimentos de matrícula "
     "com expansão de carga horária para o Período 2026.2",
     "2026-08-03", None, "matrícula", None, None, "2026.2"),
    ("Registro de matrícula no SIG@ pela SRCA: disciplinas com alteração (quebra) de pré-requisito; "
     "aluno especial (disciplinas isoladas); mobilidade acadêmica (Convênios Andifes e InterIES); "
     "Programa PEC-G",
     "2026-08-04", "2026-08-05", "matrícula", None, None, "2026.2"),
    ("Ajuste da oferta pelas Coordenações de Curso",
     "2026-08-06", "2026-08-07", "matrícula", None, None, "2026.2"),
    ("Resultado das matrículas realizadas pelos veteranos, após as 22 horas",
     "2026-08-07", None, "matrícula", None, None, "2026.2"),
    ("Matrícula de retardatários e Modificação/correção de matrícula pelos alunos para o "
     "Período 2026.2",
     "2026-08-08", "2026-08-11", "matrícula", None, None, "2026.2"),
    ("Reabertura da caderneta eletrônica referente ao Período 2026.1",
     "2026-08-08", "2026-08-17", "outro", None, None, "2026.1"),
    ("Início das aulas do Período Letivo 2026.2",
     "2026-08-10", None, "período_letivo", None, None, "2026.2"),
    ("Semana de acolhimento e integração dos discentes",
     "2026-08-10", "2026-08-14", "período_letivo", None, None, "2026.2"),
    ("Registro de matrícula pela SRCA: somente disciplinas com expansão de carga horária, conforme "
     "requerimentos aprovados e encaminhados pelas Coordenações de Curso",
     "2026-08-12", "2026-08-13", "matrícula", None, None, "2026.2"),
    ("Ajuste da oferta pelas Coordenações de Curso (após matrícula de retardatários e modificação "
     "de matrícula)",
     "2026-08-14", "2026-08-17", "matrícula", None, None, "2026.2"),
    ("Colação de Grau com solenidade dos formandos 2026.1",
     "2026-08-14", "2026-09-11", "colação", "Resolução nº 18/2017 - Conuni", None, "2026.1"),
    ("Feriado Municipal para o campus Petrolina - Padroeira de Petrolina/PE",
     "2026-08-15", None, "feriado", None, "PNZ", None),
    ("Resultado das matrículas de retardatários e modificação de matrículas realizadas pelos "
     "discentes, após as 22 horas",
     "2026-08-17", None, "matrícula", None, None, "2026.2"),
    ("Matrículas em disciplinas com vagas ociosas pelos alunos",
     "2026-08-18", "2026-08-20", "matrícula", None, None, "2026.2"),
    ("Ajuste da oferta pelas Coordenações de Curso (após a matrícula em vagas ociosas)",
     "2026-08-21", None, "matrícula", None, None, "2026.2"),
    ("Resultado das matrículas em disciplinas com vagas ociosas, após as 22 horas",
     "2026-08-21", None, "matrícula", None, None, "2026.2"),
    ("Prazo para os discentes solicitarem matrícula extemporânea às Coordenações de Curso para o "
     "Período 2026.2",
     "2026-08-24", "2026-08-28", "matrícula", "Art. 51, § 2º da Resolução nº 08/2015 - Conuni",
     None, "2026.2"),
    ("Feriado Municipal para o campus Serra da Capivara - Padroeiro de São Raimundo Nonato/PI",
     "2026-08-31", None, "feriado", None, "SRN", None),

    # ===== Setembro 2026 =====
    ("Data limite para as Coordenações de Curso realizarem as matrículas extemporâneas no SIG@ "
     "para o Período 2026.2",
     "2026-09-05", None, "matrícula", "Art. 51, § 3º da Resolução nº 08/2015 - Conuni", None, "2026.2"),
    ("Feriado Nacional para todos os campi - Independência do Brasil",
     "2026-09-07", None, "feriado", None, None, None),
    ("Trancamento do período ou cancelamento de disciplinas (com ônus), referente a 2026.2",
     "2026-09-07", "2026-09-11", "trancamento", None, None, "2026.2"),
    ("Prazo para abertura de processo de colação de grau em gabinete para os discentes que não "
     "participaram das solenidades regulares do Período 2026.1",
     "2026-09-07", "2026-09-16", "colação", "Resolução nº 18/2017 - Conuni", None, "2026.1"),
    ("Feriado Municipal para o campus Juazeiro - Padroeira de Juazeiro/BA",
     "2026-09-08", None, "feriado", None, "JUA", None),
    ("Prazo para os discentes solicitarem dispensa de disciplinas, carga horária eletiva e "
     "optativa para o Período 2027.1",
     "2026-09-17", "2026-09-30", "matrícula", None, None, "2027.1"),
    ("Feriado Municipal para o campus Petrolina - Aniversário da cidade de Petrolina/PE",
     "2026-09-21", None, "feriado", None, "PNZ", None),

    # ===== Outubro 2026 =====
    ("Colação de grau em gabinete dos formandos 2026.1 que não participaram das solenidades "
     "regulares",
     "2026-10-01", "2026-10-16", "colação", None, None, "2026.1"),
    ("Feriado Municipal para o campus Paulo Afonso - Padroeiro de Paulo Afonso/BA",
     "2026-10-04", None, "feriado", None, "PAV", None),
    ("Prazo para abertura de processo de atividades complementares para registro no Período 2026.2",
     "2026-10-05", "2026-10-16", "outro", None, None, "2026.2"),
    ("Feriado Nacional para todos os campi - Padroeira do Brasil (Nossa Senhora Aparecida)",
     "2026-10-12", None, "feriado", None, None, None),
    ("Feriado Estadual para o campus Serra da Capivara - Constituição do Piauí",
     "2026-10-19", None, "feriado", None, "SRN", None),
    ("Prazo para abertura de processos de alteração (quebra) de pré-requisito para o Período "
     "2027.1 pelo aluno ou pelos NDEs/Coordenações dos cursos de graduação",
     "2026-10-19", "2026-10-30", "outro", None, None, "2027.1"),
    ("Feriado Nacional para todos os campi - Dia do Servidor Público",
     "2026-10-28", None, "feriado", None, None, None),

    # ===== Novembro 2026 =====
    ("Feriado Nacional para todos os campi - Finados",
     "2026-11-02", None, "feriado", None, None, None),
    ("Planejamento da oferta de componentes curriculares para o Período 2027.1 pelos Colegiados "
     "Acadêmicos (19/11 é a data limite para divulgação do planejamento)",
     "2026-11-03", "2026-11-19", "planejamento", None, None, "2027.1"),
    ("Feriado Nacional para todos os campi - Proclamação da República",
     "2026-11-15", None, "feriado", None, None, None),
    ("Prazo para as Coordenações de Curso instruírem os processos de alteração (quebra) de "
     "pré-requisitos para o Período 2027.1 (parecer do docente, parecer do NDE, aprovação do "
     "Colegiado e envio à SRCA)",
     "2026-11-18", "2026-12-11", "outro", None, None, "2027.1"),
    ("Feriado Nacional para todos os campi - Dia Nacional de Zumbi e da Consciência Negra",
     "2026-11-20", None, "feriado", None, None, None),
    ("Prazo para solicitação de matrícula especial/disciplinas isoladas (estudantes de outras "
     "IES) para o Período 2027.1",
     "2026-11-23", "2026-11-27", "matrícula", None, None, "2027.1"),
    ("Scientex - Dias letivos a serem registrados no diário de classe como atividades de ensino "
     "(carga horária) aos que participarem do evento (aulas suspensas)",
     "2026-11-24", "2026-11-27", "outro", None, None, None),
    ("Data limite para as Coordenações de Curso enviarem os processos de atividades "
     "complementares para registro no Período 2027.1",
     "2026-11-27", None, "outro", None, None, "2027.1"),

    # ===== Dezembro 2026 =====
    ("Oferta de componentes curriculares para o Período 2027.1 pelas Coordenações de Curso "
     "no SIG@",
     "2026-12-01", "2026-12-11", "planejamento", None, None, "2027.1"),
    ("Último dia de aulas do Período 2026.2",
     "2026-12-12", None, "período_letivo", None, None, "2026.2"),
    ("Feriado Municipal para o campus Serra da Capivara - Festividade de Santa Luzia em São "
     "Raimundo Nonato/PI",
     "2026-12-13", None, "feriado", None, "SRN", None),
    ("Início do prazo para preenchimento dos planos de ensino na caderneta eletrônica para o "
     "Período 2027.1",
     "2026-12-14", None, "outro", None, None, "2027.1"),
    ("Exames Finais do Período 2026.2",
     "2026-12-14", "2026-12-18", "exames", None, None, "2026.2"),
    ("Prazo para os discentes solicitarem matrícula com expansão de carga horária para o "
     "Período 2027.1 às Coordenações de Curso",
     "2026-12-14", "2026-12-18", "matrícula", None, None, "2027.1"),
    ("Feriado Municipal para o campus Salgueiro - Aniversário da Cidade de Salgueiro/PE",
     "2026-12-23", None, "feriado", None, "SAL", None),
    ("Data limite para as Coordenações de Curso encaminharem à SRCA os processos de dispensa de "
     "disciplinas, alteração (quebra) de pré-requisitos e matrícula especial/disciplinas isoladas "
     "(estudantes de outras IES) para o Período 2027.1",
     "2026-12-22", None, "outro", None, None, "2027.1"),
    ("Recesso acadêmico e administrativo - Feriado Nacional - Natal",
     "2026-12-24", "2026-12-25", "feriado", None, None, None),
    ("Último dia para lançamento das notas e faltas do Período 2026.2 pelos docentes. Data limite "
     "para finalizar preenchimento da caderneta eletrônica referente a 2026.2",
     "2026-12-30", None, "outro", "Resolução nº 16/2021 - Conuni", None, "2026.2"),
    ("Encerramento do Período Letivo 2026.2",
     "2026-12-30", None, "período_letivo", None, None, "2026.2"),
    ("Recesso acadêmico e administrativo - Feriado Nacional - Confraternização Universal",
     "2026-12-31", "2027-01-02", "feriado", None, None, None),

    # ===== Janeiro 2027 =====
    ("Feriado Municipal para o campus Senhor do Bonfim - Padroeiro de Senhor do Bonfim/BA",
     "2027-01-17", None, "feriado", None, "SBF", None),

    # ===== Fevereiro 2027 =====
    ("Colação de Grau com solenidade dos formandos 2026.2",
     "2027-02-01", "2027-02-26", "colação", "Resolução nº 18/2017 - Conuni", None, "2026.2"),
    ("Recesso acadêmico e administrativo - Ponto Facultativo - Carnaval/Quarta-feira de Cinzas",
     "2027-02-08", "2027-02-10", "feriado", None, None, None),
    ("Prazo para abertura de processo de colação de grau em gabinete para os discentes que não "
     "participaram das solenidades regulares do Período 2026.2",
     "2027-02-22", "2027-03-03", "colação", "Resolução nº 18/2017 - Conuni", None, "2026.2"),

    # ===== Março 2027 =====
    ("Feriado Estadual para os campi no Estado de Pernambuco - Data Magna de Pernambuco",
     "2027-03-06", None, "feriado", None, None, None),
    ("Provável início das aulas do Período Letivo 2027.1 (data a confirmar com base no "
     "cronograma do Sisu)",
     "2027-03-08", None, "período_letivo", None, None, "2027.1"),
    ("Colação de grau em gabinete dos formandos 2026.2 que não participaram das solenidades "
     "regulares",
     "2027-03-15", "2027-03-31", "colação", None, None, "2026.2"),
]


def main():
    logger.info("=" * 60)
    logger.info("SEED — Calendário Acadêmico de Graduação 2026")
    logger.info("=" * 60)

    session = SessionLocal()
    try:
        created, skipped = 0, 0
        for title, start, end, category, legal_ref, campus, period in EVENTS:
            start_date = _d(start)
            existing = (
                session.query(AcademicEvent)
                .filter_by(title=title, start_date=start_date)
                .one_or_none()
            )
            if existing:
                skipped += 1
                continue

            event = AcademicEvent(
                course_id=None,
                title=title,
                start_date=start_date,
                end_date=_d(end),
                category=category,
                legal_reference=legal_ref,
                campus=campus,
                academic_period=period,
                source="seed",
            )
            session.add(event)
            created += 1
            logger.info(f"  + {start} — {title[:70]}")

        session.commit()
        logger.info(f"\nSeed concluído — {created} criados, {skipped} já existiam.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
