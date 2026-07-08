"""
CalendarTool — consulta estruturada sobre o calendário acadêmico (SQL, não RAG)

Mesma lógica do DisciplineTool/ProfessorTool (fecha D9): prazos exatos
(período de trancamento de matrícula, início/fim de período letivo, feriados
por campus) são fatos exatos — não devem depender de recall de embedding
sobre o PDF do calendário.
"""

from __future__ import annotations

from src.calendar_events import service as calendar_service

NAME = "search_academic_calendar"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Consulta o calendário acadêmico oficial — datas exatas de início/fim de "
            "período letivo, prazos de matrícula/rematrícula/trancamento, feriados, "
            "colação de grau, exames finais e demais prazos administrativos. Use "
            "SEMPRE que a pergunta envolver 'quando', 'até que dia', ou prazos de "
            "eventos acadêmicos — não tente responder isso só com "
            "search_normative_documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Categoria do evento (ex: 'matrícula', 'trancamento', 'feriado', "
                        "'colação', 'período_letivo', 'exames'). Omitir para não filtrar."
                    ),
                },
                "academic_period": {
                    "type": "string",
                    "description": "Período letivo no formato 'AAAA.N' (ex: '2026.1'). Omitir para não filtrar.",
                },
            },
            "required": [],
        },
    },
}


def execute(arguments: dict, context: dict) -> dict:
    db = context["db"]
    events = calendar_service.list_events(
        db,
        course_id=context.get("course_id"),
        category=arguments.get("category"),
        academic_period=arguments.get("academic_period"),
    )

    if not events:
        return {"summary": "Nenhum evento encontrado para esses critérios.", "sources": []}

    sources: list[dict] = []
    lines: list[str] = []
    for e in events:
        date_range = e.start_date.isoformat()
        if e.end_date and e.end_date != e.start_date:
            date_range += f" a {e.end_date.isoformat()}"
        campus = f" | campus: {e.campus}" if e.campus else ""
        legal = f" | {e.legal_reference}" if e.legal_reference else ""
        lines.append(f"- {e.title} | {date_range}{campus}{legal}")
        sources.append(
            {
                "origin": "calendar",
                "title": e.title,
                "start_date": e.start_date.isoformat(),
                "end_date": e.end_date.isoformat() if e.end_date else "",
                "category": e.category or "",
                "legal_reference": e.legal_reference or "",
                "campus": e.campus or "",
                "academic_period": e.academic_period or "",
            }
        )

    return {"summary": "\n".join(lines), "sources": sources}
