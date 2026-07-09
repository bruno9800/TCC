"""
Schemas de Extração Estruturada — Importação via LLM

Modelos Pydantic usados como `response_format` nas chamadas de Structured
Outputs da OpenAI (client.chat.completions.parse). Cada schema espelha o
formato das tuplas dos scripts de seed (scripts/seed_calendar_2026.py,
scripts/seed_disciplines_engcomp.py) — o pipeline automatiza exatamente o
processo manual que construiu aqueles seeds.

Datas são extraídas como string ISO (e não `date`) de propósito: a validação
é feita depois, no importador, para que um item malformado vire um warning no
preview em vez de derrubar a extração inteira.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Mesmos valores usados no seed e filtrados pela calendar_tool do agente.
EventCategory = Literal[
    "feriado",
    "matrícula",
    "trancamento",
    "colação",
    "exames",
    "período_letivo",
    "planejamento",
    "outro",
]

# Códigos de campus usados no seed (título do evento menciona a cidade).
CampusCode = Literal["JUA", "PNZ", "PAV", "SAL", "SBF", "SRN"]


# ── Calendário Acadêmico ─────────────────────────────────────────────────────


class ExtractedEvent(BaseModel):
    title: str
    start_date: str  # ISO "YYYY-MM-DD"
    end_date: str | None
    category: EventCategory
    legal_reference: str | None
    campus: CampusCode | None
    academic_period: str | None  # "AAAA.N", ex: "2026.1"


class ExtractedEventList(BaseModel):
    events: list[ExtractedEvent]


# ── Matriz Curricular (PPC) ──────────────────────────────────────────────────


class ExtractedDiscipline(BaseModel):
    name: str
    code: str | None  # sigla, ex: "APC"
    period: int | None  # 1-10; None para optativas
    workload: int | None  # carga horária em horas
    prerequisites_text: str | None  # ex: "Pré-requisito: APC; Co-requisito: RC"


class ExtractedDisciplineList(BaseModel):
    disciplines: list[ExtractedDiscipline]


# ── Itinerário do Transporte Estudantil ──────────────────────────────────────


class ExtractedStop(BaseModel):
    time: str | None  # "HH:MM"; None para sub-cabeçalhos (ex: "SEGUNDA VIAGEM...")
    location: str


class ExtractedRoute(BaseModel):
    bus_label: str  # ex: "A"
    route_description: str  # ex: "JUAZEIRO -> CCA -> COHAB -> CCA"
    shift: Literal["manhã", "tarde", "noite"]
    stops: list[ExtractedStop]
