"""
RoutesImporter — PDF do Itinerário do Transporte Estudantil (PROAE) → transport_routes

O PDF é uma sequência de tabelas, uma por ônibus, agrupadas por turno
("Itinerário MANHÃ", "ITINERÁRIO TARDE - RETORNO...", "Itinerário NOITE...").
Cada tabela tem linhas (horário, ponto de embarque/desembarque). A entrada é
o TEXTO PURO do PDF (extractor.pdf_bytes_to_text), não o Markdown: neste
documento a detecção de tabelas do pymupdf4llm descarta tabelas inteiras e
duplica linhas. O fatiamento é pelo cabeçalho de ônibus (ÔNIBUS "A"...),
carregando o cabeçalho de turno mais recente como contexto — a palavra
"ÔNIBUS" também aparece em nomes de parada ("PONTO DE ÔNIBUS..."), então o
regex exige a letra da linha entre aspas, como só ocorre nos cabeçalhos.

Semântica do apply (replace-by-semester): apaga todas as rotas do semestre
informado e insere as extraídas. O semestre é parâmetro do admin no upload
(ex: "2026.1") — o PDF não o declara de forma confiável.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

from sqlalchemy.orm import Session

from src.db.models import TransportRoute, TransportRouteStop
from src.imports.extractor import parse_sections
from src.imports.schemas import ExtractedRoute

logger = logging.getLogger(__name__)

# ÔNIBUS "A" / ÔNIBUS “B” — aspas retas ou curvas em volta da letra da linha.
_BUS_HEADER_RE = re.compile(r"ÔNIBUS\s*[\"“”']\s*([A-Z])\s*[\"“”']", re.IGNORECASE)
_SHIFT_HEADER_RE = re.compile(r"Itiner[áa]rio[^|\n]*", re.IGNORECASE)
_EFFECTIVE_DATE_RE = re.compile(r"A\s+PARTIR\s+DE\s+(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
_TIME_RE = re.compile(r"^(\d{1,2})\s*[:;hH.]\s*(\d{2})$")

SYSTEM_PROMPT = """\
Você extrai UMA rota do itinerário do transporte estudantil da UNIVASF (PROAE) a \
partir da tabela de um ônibus (convertida de PDF para Markdown).

- bus_label: a letra da linha ("A", "B"...). Anote só a letra, sem aspas.
- route_description: o trajeto do cabeçalho (ex: "JUAZEIRO -> CCA -> COHAB -> CCA"). \
Se o cabeçalho tiver observações como "(Micro)" ou "SAÍDA DO CCA ÀS 15:10", inclua-as.
- shift: "manhã", "tarde" ou "noite" — deduza do cabeçalho de turno informado no \
início da mensagem (ex: "ITINERÁRIO 14:10..." é tarde; saídas 18:30+ do turno \
noturno são "noite").
- stops: TODAS as paradas na ordem da tabela, cada uma com horário "HH:MM" \
(normalize erros de digitação como "13;15" ou "14: 25") e o nome do ponto limpo \
de marcações Markdown. Linhas de sub-cabeçalho sem horário (ex: "SEGUNDA VIAGEM \
(Saindo do CCA)...") entram como parada com time null. Ignore linhas vazias e as \
linhas "HORÁRIO | PONTOS DE EMBARQUE...".
Não invente paradas nem horários."""


def _split_by_bus(md: str) -> list[tuple[str, str]]:
    """Fatia o markdown em [(rótulo 'ÔNIBUS A (turno...)', chunk)] por cabeçalho de ônibus."""
    matches = list(_BUS_HEADER_RE.finditer(md))
    sections: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
        # cabeçalho de turno mais recente antes deste ônibus
        shift_headers = _SHIFT_HEADER_RE.findall(md[: m.start()])
        shift_context = shift_headers[-1].strip() if shift_headers else "não informado"
        chunk = (
            f"Cabeçalho de turno mais recente no documento: {shift_context}\n\n"
            + md[m.start() : end]
        )
        sections.append((f"ÔNIBUS {m.group(1).upper()} #{idx + 1}", chunk))
    return sections


def _normalize_time(raw: str | None, label: str, warnings: list[str]) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    # o LLM às vezes devolve o literal "null" como string em vez de JSON null
    if not raw or raw.lower() in {"null", "none", "-", "--"}:
        return None
    m = _TIME_RE.match(raw)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    warnings.append(f"Seção '{label}': horário não reconhecido '{raw}' — mantido como está")
    return raw[:10]


def extract(text: str) -> tuple[list[dict], list[str]]:
    """Extrai as rotas do texto puro do itinerário (pdf_bytes_to_text)."""
    sections = _split_by_bus(text)
    if not sections:
        return [], ["Nenhum cabeçalho de ônibus (ÔNIBUS \"A\"...) encontrado no PDF — é o itinerário do transporte?"]

    effective_date: str | None = None
    m = _EFFECTIVE_DATE_RE.search(text)
    if m:
        effective_date = datetime.strptime(m.group(1), "%d/%m/%Y").date().isoformat()

    results, warnings = parse_sections(SYSTEM_PROMPT, sections, ExtractedRoute)

    items: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()
    for label, route in results:
        stops = [
            {
                "time": _normalize_time(s.time, label, warnings),
                "location": re.sub(r"\s+", " ", s.location).strip(),
            }
            for s in route.stops
            if s.location and s.location.strip()
        ]
        if not stops:
            warnings.append(f"Seção '{label}': rota sem paradas — descartada")
            continue

        departure = next((s["time"] for s in stops if s["time"]), None)
        key = (route.shift, route.bus_label.strip().upper(), departure)
        if key in seen:
            warnings.append(f"Seção '{label}': rota duplicada ({key}) — mantida a primeira")
            continue
        seen.add(key)

        items.append(
            {
                "bus_label": route.bus_label.strip().upper(),
                "route_description": re.sub(r"\s+", " ", route.route_description).strip(),
                "shift": route.shift,
                "effective_date": effective_date,
                "stops": stops,
            }
        )

    if not items:
        warnings.append("Nenhuma rota extraída do PDF.")
    return items, warnings


def diff(db: Session, items: list[dict], semester: str) -> dict:
    """Compara com o estado atual: replace integral por semestre."""
    existing = db.query(TransportRoute).filter_by(semester=semester).count()
    return {
        "semester": semester,
        "extracted": len(items),
        "removed": existing,
        "added": len(items),
        "total_stops": sum(len(i["stops"]) for i in items),
    }


def apply(db: Session, items: list[dict], semester: str) -> dict:
    """Replace-by-semester: substitui todas as rotas do semestre."""
    if not items:
        raise ValueError("Nada a aplicar: payload vazio")

    old_routes = db.query(TransportRoute).filter_by(semester=semester).all()
    for route in old_routes:  # delete via ORM para o cascade remover as paradas
        db.delete(route)

    for item in items:
        route = TransportRoute(
            semester=semester,
            shift=item["shift"],
            bus_label=item["bus_label"],
            route_description=item["route_description"],
            section_title=None,
            effective_date=(
                date.fromisoformat(item["effective_date"]) if item.get("effective_date") else None
            ),
        )
        route.stops = [
            TransportRouteStop(seq=i, time=s["time"], location=s["location"])
            for i, s in enumerate(item["stops"], start=1)
        ]
        db.add(route)

    return {"deleted": len(old_routes), "inserted": len(items)}
