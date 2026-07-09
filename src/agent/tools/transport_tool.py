"""
TransportTool — consulta estruturada sobre o transporte estudantil (SQL, não RAG)

Mesma lógica do CalendarTool (D9): horários e pontos de parada dos ônibus da
PROAE são fatos exatos — não devem depender de recall de embedding sobre o
PDF do itinerário.
"""

from __future__ import annotations

from src.transport import service as transport_service

NAME = "search_transport_routes"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Consulta o itinerário oficial do transporte estudantil da UNIVASF (ônibus "
            "da PROAE entre os campi Juazeiro, Petrolina e Ciências Agrárias/CCA): "
            "linhas, horários e pontos de embarque/desembarque por turno. Use SEMPRE "
            "que a pergunta envolver ônibus, transporte, itinerário, ou 'que horas "
            "passa em' determinado ponto."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "shift": {
                    "type": "string",
                    "enum": ["manhã", "tarde", "noite"],
                    "description": "Turno do itinerário. Omitir para todos os turnos.",
                },
                "location": {
                    "type": "string",
                    "description": (
                        "Filtra rotas que passam por um ponto (busca parcial, ex: "
                        "'COHAB', 'Residência Estudantil', 'Campus Petrolina'). "
                        "Omitir para não filtrar."
                    ),
                },
                "semester": {
                    "type": "string",
                    "description": "Semestre no formato 'AAAA.N' (ex: '2026.1'). Omitir para o mais recente.",
                },
            },
            "required": [],
        },
    },
}


def execute(arguments: dict, context: dict) -> dict:
    db = context["db"]
    location = arguments.get("location")
    routes = transport_service.list_routes(
        db,
        semester=arguments.get("semester"),
        shift=arguments.get("shift"),
        location=location,
    )

    if not routes:
        return {
            "summary": "Nenhuma rota de transporte encontrada para esses critérios.",
            "sources": [],
        }

    sources: list[dict] = []
    lines: list[str] = []
    for r in routes:
        timed = [s for s in r.stops if s.time]
        first, last = (timed[0], timed[-1]) if timed else (None, None)
        header = f"Ônibus {r.bus_label} ({r.shift}, {r.semester}) — {r.route_description}"
        if first and last:
            header += f" | {first.time} ({first.location}) → {last.time} ({last.location})"
        lines.append(f"- {header}")

        if location:
            for s in timed:
                if location.strip().lower() in s.location.lower():
                    lines.append(f"    passa às {s.time} em {s.location}")
        elif len(routes) <= 3:
            for s in r.stops:
                lines.append(f"    {s.time or '--:--'} {s.location}")

        sources.append(
            {
                "origin": "transport",
                "bus_label": r.bus_label,
                "shift": r.shift,
                "semester": r.semester,
                "route_description": r.route_description,
                "effective_date": r.effective_date.isoformat() if r.effective_date else "",
            }
        )

    return {"summary": "\n".join(lines), "sources": sources}
