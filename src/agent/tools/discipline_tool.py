"""
DisciplineTool — consulta estruturada sobre a matriz curricular (SQL, não RAG)

Mesma lógica do ProfessorTool (fecha D9): período, carga horária e
pré-requisitos de uma disciplina são fatos exatos — não devem depender de
recall de embedding sobre o Ementário em prosa do PPC.

Achado real durante a verificação da Fase 5a: perguntado sobre os
pré-requisitos de "Compiladores" usando só RAG, o agente respondeu "Teoria
da Computação e Estruturas de Dados" — a resposta correta, segundo a matriz
curricular oficial, é "AED, LFA e OAC". Esta tool wrap fino sobre
`professors.service.list_disciplines` (já existe desde a Fase 3) elimina
essa classe de erro para perguntas sobre a matriz curricular.
"""

from __future__ import annotations

from src.professors import service as professor_service

NAME = "search_disciplines"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Consulta a matriz curricular oficial do curso — período, carga horária, "
            "código e pré-requisitos/co-requisitos exatos de uma disciplina. Use para "
            "perguntas sobre em que período fica uma disciplina, quais os "
            "pré-requisitos, ou a carga horária de uma disciplina específica."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome (ou parte do nome) da disciplina.",
                },
                "period": {
                    "type": "integer",
                    "description": "Número do período (1 a 10). Omitir para disciplinas optativas.",
                },
            },
            "required": [],
        },
    },
}


def execute(arguments: dict, context: dict) -> dict:
    db = context["db"]
    disciplines = professor_service.list_disciplines(db, course_id=context.get("course_id"))

    name_filter = (arguments.get("name") or "").strip().lower()
    period_filter = arguments.get("period")

    if name_filter:
        disciplines = [d for d in disciplines if name_filter in d.name.lower()]
    if period_filter is not None:
        disciplines = [d for d in disciplines if d.period == period_filter]

    if not disciplines:
        return {"summary": "Nenhuma disciplina encontrada para esses critérios.", "sources": []}

    sources: list[dict] = []
    lines: list[str] = []
    for d in disciplines:
        prereq = f" | {d.prerequisites_text}" if d.prerequisites_text else ""
        lines.append(
            f"- {d.name} ({d.code}) | período: {d.period or 'optativa'} | "
            f"carga horária: {d.workload}h{prereq}"
        )
        sources.append(
            {
                "origin": "discipline",
                "name": d.name,
                "code": d.code or "",
                "period": d.period,
                "workload": d.workload,
                "prerequisites_text": d.prerequisites_text or "",
            }
        )

    return {"summary": "\n".join(lines), "sources": sources}
