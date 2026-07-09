"""
CalendarImporter — PDF do Calendário Acadêmico → academic_events

Automatiza o processo manual que construiu scripts/seed_calendar_2026.py.
O PDF é organizado em blocos mensais (grade do mês + tabela "DIAS |
ATIVIDADES") onde as datas aparecem sem ano ("31/12 a 02/01", "15 a 22") — o
ano vem do cabeçalho do bloco ("JANEIRO 2026"). Por isso o fatiamento é por
mês: cada bloco vira uma chamada de extração com o mês/ano de referência
explícitos no prompt.

Semântica do apply (replace-by-scope): apaga eventos não-manuais cuja
start_date caia na janela coberta pelo novo PDF e insere os extraídos com
source='import'. Reimportar o mesmo PDF é idempotente; eventos criados à mão
pelo admin (source='manual') sobrevivem.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from sqlalchemy.orm import Session

from src.db.models import AcademicEvent
from src.imports.extractor import parse_sections
from src.imports.schemas import ExtractedEventList

logger = logging.getLogger(__name__)

MONTHS_PT = [
    "JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO",
    "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO",
]

# "JANEIRO 2026", "MARÇO DE 2027" — exige o ano na sequência para não casar
# com menções soltas a meses dentro do texto de um evento.
_MONTH_HEADER_RE = re.compile(
    r"\b(" + "|".join(MONTHS_PT) + r")\s*(?:DE\s+)?(20\d{2})\b", re.IGNORECASE
)

SYSTEM_PROMPT = """\
Você extrai eventos do Calendário Acadêmico de Graduação da UNIVASF a partir de \
uma seção mensal do documento (convertido de PDF para Markdown).

A seção contém uma grade do mês (dias da semana — ignore) e uma tabela de \
atividades com linhas no formato "DIAS | ATIVIDADE". Extraia UM evento por linha \
de atividade. Ignore linhas de contagem ("Dias Letivos", "Acumulado", "Subtotal") \
e a grade do calendário.

## Datas
- As datas vêm sem ano ("15 a 22", "31/12 a 02/01"). Use o mês/ano de referência \
informados no início da mensagem. Quando a linha indicar explicitamente outro mês \
("31/12 a 02/01" numa seção de janeiro), resolva para o mês indicado no próprio \
texto, ajustando o ano quando o intervalo cruza a virada (ex.: seção JANEIRO/2026, \
"31/12 a 02/01" → start 2025-12-31, end 2026-01-02).
- Intervalo "15 a 22" → start_date e end_date. Dia único → end_date null.

## Categoria (escolha exatamente uma)
- "feriado": feriados, recessos, pontos facultativos.
- "matrícula": tudo que faz parte da operação de matrícula — matrícula/rematrícula, \
"Preparação para a matrícula (SIG@ indisponível)", "Ajuste da oferta pelas \
Coordenações", "Resultado das matrículas", registro de matrícula pela SRCA, \
reintegração, dispensa de disciplinas, matrícula especial/extemporânea/vagas ociosas.
- "trancamento": trancamento de período ou cancelamento de disciplinas.
- "colação": colação de grau (solenidade ou gabinete).
- "exames": exames finais.
- "período_letivo": início/fim de aulas, encerramento de período, semana de acolhimento.
- "planejamento": planejamento/oferta de componentes curriculares, divulgação de \
programas pelos Colegiados.
- "outro": o que não couber acima (caderneta eletrônica, atividades complementares, \
quebra de pré-requisito, eventos como Scientex etc.).

## Campus
Preencha apenas se o evento for específico de um campus: Juazeiro → "JUA", \
Petrolina → "PNZ", Paulo Afonso → "PAV", Salgueiro → "SAL", Senhor do Bonfim → \
"SBF", São Raimundo Nonato / Serra da Capivara → "SRN". Feriados estaduais que \
valem para vários campi ao mesmo tempo → null.

## Outros campos
- title: texto completo da atividade, limpo de marcações Markdown (** etc.), sem \
abreviar.
- legal_reference: apenas se a linha citar norma ("Art. X da Resolução Y") — senão null.
- academic_period: o período letivo a que o evento se REFERE, copiado da menção \
explícita no texto ("para o Período 2026.1" → "2026.1"; "formandos 2025.2" → \
"2025.2"; "referente a 2026.1" → "2026.1") — NÃO o período em que o evento \
acontece no tempo. Feriados e eventos sem menção a período → null.
Não invente eventos: extraia somente o que estiver nas linhas de atividade."""


def split_by_month(md: str) -> list[tuple[str, str]]:
    """Fatia o markdown em blocos mensais [(rótulo 'JANEIRO/2026', texto), ...]."""
    matches = list(_MONTH_HEADER_RE.finditer(md))
    if not matches:
        return []

    # Vários "JANEIRO 2026" podem aparecer no mesmo bloco (grade + rodapés);
    # agrupa ocorrências consecutivas do mesmo mês/ano num bloco só.
    sections: list[tuple[str, str]] = []
    current_key: str | None = None
    current_start = 0
    for m in matches:
        key = f"{m.group(1).upper()}/{m.group(2)}"
        if key != current_key:
            if current_key is not None:
                sections.append((current_key, md[current_start : m.start()]))
            current_key = key
            current_start = m.start()
    sections.append((current_key, md[current_start:]))
    return sections


def _parse_iso(value: str | None, label: str, field: str, warnings: list[str]) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        warnings.append(f"Seção '{label}': {field} inválida '{value}' — item descartado")
        return None


def extract(md: str) -> tuple[list[dict], list[str]]:
    """
    Extrai eventos do markdown do calendário.

    Returns:
        (itens normalizados prontos para o payload do ImportJob, warnings)
    """
    sections = split_by_month(md)
    if not sections:
        return [], ["Nenhum cabeçalho de mês (ex: 'JANEIRO 2026') encontrado no PDF — é o calendário acadêmico?"]

    prompted = [
        (label, f"Mês/ano de referência desta seção: {label}\n\n{content}")
        for label, content in sections
    ]
    results, warnings = parse_sections(SYSTEM_PROMPT, prompted, ExtractedEventList)

    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for label, parsed in results:
        for event in parsed.events:
            start = _parse_iso(event.start_date, label, "start_date", warnings)
            if start is None:
                continue
            end = _parse_iso(event.end_date, label, "end_date", warnings) if event.end_date else None
            if end is not None and end < start:
                warnings.append(
                    f"Seção '{label}': '{event.title[:60]}' com end_date < start_date — end_date descartado"
                )
                end = None

            title = re.sub(r"\s+", " ", event.title).strip()
            key = (title.lower(), start.isoformat())
            if key in seen:  # mesmo evento repetido em blocos vizinhos
                continue
            seen.add(key)

            items.append(
                {
                    "title": title,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat() if end else None,
                    "category": event.category,
                    "legal_reference": event.legal_reference,
                    "campus": event.campus,
                    "academic_period": event.academic_period,
                }
            )

    if not items:
        warnings.append("Nenhum evento extraído do PDF.")
    return items, warnings


def _scope(items: list[dict]) -> tuple[date, date]:
    dates = [date.fromisoformat(i["start_date"]) for i in items]
    return min(dates), max(dates)


def diff(db: Session, items: list[dict]) -> dict:
    """Compara os eventos extraídos com o estado atual do banco (janela do PDF)."""
    if not items:
        return {"added": 0, "unchanged": 0, "replaced": 0, "kept_manual": 0}

    lo, hi = _scope(items)
    existing = (
        db.query(AcademicEvent)
        .filter(AcademicEvent.start_date >= lo, AcademicEvent.start_date <= hi)
        .all()
    )
    existing_keys = {
        (e.title.lower(), e.start_date.isoformat()) for e in existing if e.source != "manual"
    }
    new_keys = {(i["title"].lower(), i["start_date"]) for i in items}

    manual = [e for e in existing if e.source == "manual"]
    return {
        "scope_start": lo.isoformat(),
        "scope_end": hi.isoformat(),
        "extracted": len(items),
        "added": len(new_keys - existing_keys),
        "unchanged": len(new_keys & existing_keys),
        "removed": len(existing_keys - new_keys),
        "kept_manual": len(manual),
    }


def apply(db: Session, items: list[dict]) -> dict:
    """Replace-by-scope: substitui os eventos não-manuais da janela do PDF."""
    if not items:
        raise ValueError("Nada a aplicar: payload vazio")

    lo, hi = _scope(items)
    deleted = (
        db.query(AcademicEvent)
        .filter(
            AcademicEvent.start_date >= lo,
            AcademicEvent.start_date <= hi,
            AcademicEvent.source != "manual",
        )
        .delete(synchronize_session=False)
    )

    for item in items:
        db.add(
            AcademicEvent(
                course_id=None,  # calendário acadêmico vale para todos os cursos
                title=item["title"],
                start_date=date.fromisoformat(item["start_date"]),
                end_date=date.fromisoformat(item["end_date"]) if item["end_date"] else None,
                category=item["category"],
                legal_reference=item["legal_reference"],
                campus=item["campus"],
                academic_period=item["academic_period"],
                source="import",
            )
        )

    return {"deleted": deleted, "inserted": len(items)}
