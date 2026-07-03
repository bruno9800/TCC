"""
ProfessorTool — consulta estruturada sobre o corpo docente (SQL, não RAG)

Fecha a decisão D9 (EVOLUTION.md): fatos exatos (nome, e-mail, área) não
devem depender de recall de embedding. Wrap fino sobre ProfessorService
(Fase 3) — nenhuma lógica de consulta é reimplementada aqui.
"""

from __future__ import annotations

from src.professors import service as professor_service

NAME = "search_professors"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Consulta o corpo docente da UNIVASF — nome, e-mail, área de atuação, Lattes, "
            "site pessoal, participação no Núcleo Docente Estruturante (NDE). Use para "
            "perguntas sobre professores específicos ou sobre quem atua em determinada área."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome (ou parte do nome) do professor.",
                },
                "area": {
                    "type": "string",
                    "description": "Área de atuação ou disciplina (ex: 'Robótica', 'Bancos de Dados').",
                },
            },
            "required": [],
        },
    },
}


def execute(arguments: dict, context: dict) -> dict:
    db = context["db"]
    professors = professor_service.list_professors(
        db,
        course_id=context.get("course_id"),
        area=arguments.get("area"),
        name=arguments.get("name"),
    )

    if not professors:
        return {"summary": "Nenhum professor encontrado para esses critérios.", "sources": []}

    sources: list[dict] = []
    lines: list[str] = []
    for p in professors:
        if p.nde_role:
            nde_tag = f" [NDE — {p.nde_role}]"
        elif p.is_nde:
            nde_tag = " [NDE]"
        else:
            nde_tag = ""
        lines.append(f"- {p.name} | e-mail: {p.email} | área: {p.area or 'não informada'}{nde_tag}")
        sources.append(
            {
                "origin": "professor",
                "name": p.name,
                "email": p.email,
                "area": p.area or "",
                "department": p.department or "",
                "is_nde": p.is_nde,
                "nde_role": p.nde_role or "",
                "lattes_url": p.lattes_url or "",
            }
        )

    return {"summary": "\n".join(lines), "sources": sources}
