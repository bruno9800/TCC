"""
CurriculumImporter — PDF do PPC → disciplines

Automatiza o processo manual que construiu scripts/seed_disciplines_engcomp.py.
Em vez de depender da numeração de seção de um PPC específico (4.2, 4.2.12...),
o fatiamento localiza as tabelas de matriz curricular pela estrutura: blocos de
tabela Markdown cujo cabeçalho contém "Disciplina". Cada tabela vai ao LLM com
o título da seção imediatamente anterior (que informa o período — "Matriz
curricular do 3º período" — ou que são optativas).

Semântica do apply (upsert por code): atualiza/insere por (course_id, code).
Disciplinas que existem no banco mas sumiram do novo PPC não são apagadas às
cegas — só as sem vínculo com professores (professor_disciplines); as demais
ficam e aparecem como warning no preview.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy.orm import Session

from src.db.models import Discipline, ProfessorDiscipline
from src.imports.extractor import parse_sections
from src.imports.schemas import ExtractedDisciplineList

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Você extrai a matriz curricular de um PPC (Projeto Pedagógico de Curso) da \
UNIVASF a partir de tabelas Markdown com colunas como \
Disciplina | Sigla | Pré-requisito | Co-requisito | CH.

Para cada linha de disciplina extraia:
- name: nome completo da disciplina, limpo de marcações Markdown. A conversão do \
PDF perde ligaduras tipográficas ("fi", "fl") e pode quebrar a linha no meio do \
nome — repare esses casos (ex: "Computação Grá\\nca" → "Computação Gráfica", \
"Inteligência Articial" → "Inteligência Artificial", "Criptograa" → "Criptografia").
- code: a sigla (ex: "APC", "CDI-I"). Se não houver, null.
- period: o número do período indicado no TÍTULO DA SEÇÃO acima da tabela \
("Matriz curricular do 3º período" → 3). Somente quando o título da seção for de \
disciplinas OPTATIVAS ("Disciplinas Optativas - ...") → null. Linhas como \
"Optativa I" ou "Eletiva II" DENTRO de uma tabela de período são vagas da grade \
daquele período: mantenha o period da seção e a sigla da linha (ex: "Optativa IV" \
com sigla OPT-IV no 10º período → period 10, code "OPT-IV").
- workload: a carga horária (CH) em horas, como inteiro.
- prerequisites_text: monte a partir das colunas de pré e co-requisito, no formato \
"Pré-requisito: X" / "Co-requisito: Y" / "Pré-requisito: X; Co-requisito: Y". \
Colunas com "-" ou vazias não entram. Se nenhum dos dois existir → null. Preserve \
o texto original dos requisitos (ex: "AL, CDI-II e APC", "70% do curso"). ATENÇÃO: \
as colunas seguem a ordem do cabeçalho (Disciplina | Sigla | Pré-requisito | \
Co-requisito | CH) — confira célula a célula; a coluna Co-requisito com qualquer \
sigla diferente de "-" é um co-requisito real e DEVE entrar em prerequisites_text \
(ex: linha "Banco de Dados I | BD-I | - | ES-I | 60" → "Co-requisito: ES-I").

Ignore linhas de "Total", cabeçalhos repetidos e qualquer linha que não seja uma \
disciplina. Não invente disciplinas."""


def _table_blocks(md: str) -> list[tuple[str, str]]:
    """
    Localiza blocos de tabela Markdown cujo cabeçalho contém "Disciplina" e
    "Sigla" (a matriz curricular; tabelas de ementas têm só "Disciplina"),
    devolvendo cada um com o título de seção imediatamente anterior.

    Uma linha de tabela pode NÃO começar com "|": ligaduras tipográficas do
    PDF ("fi" em "Gráfica") viram quebra de linha no meio da célula na
    conversão ("|Computação Grá" / "ca<br>|CG|..."), então a continuação é
    qualquer linha que ainda contenha "|".
    """
    lines = md.splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        header = lines[i].lstrip()
        if header.startswith("|") and "disciplina" in header.lower() and "sigla" in header.lower():
            start = i
            while i < len(lines) and ("|" in lines[i] or not lines[i].strip()):
                if not lines[i].strip() and (i + 1 >= len(lines) or "|" not in lines[i + 1]):
                    break
                i += 1
            # título da seção: última linha não vazia antes da tabela
            heading = ""
            for j in range(start - 1, max(start - 6, -1), -1):
                if lines[j].strip():
                    heading = lines[j].strip()
                    break
            table = "\n".join(lines[start:i])
            blocks.append((heading, f"{heading}\n\n{table}"))
        else:
            i += 1
    return blocks


def extract(md: str) -> tuple[list[dict], list[str]]:
    """
    Extrai as disciplinas da matriz curricular do markdown do PPC.

    Uma chamada de LLM por tabela (não em lote): com várias tabelas no mesmo
    prompt o modelo passou a perder células — o co-requisito de BD-I sumiu em
    todo lote testado, mas nunca com a tabela isolada. As chamadas rodam em
    paralelo (parse_sections), então o custo em latência é pequeno.
    """
    blocks = _table_blocks(md)
    if not blocks:
        return [], [
            "Nenhuma tabela de matriz curricular (cabeçalho com 'Disciplina') encontrada no PDF — é o PPC?"
        ]

    sections = [(heading[:60] or "tabela", content) for heading, content in blocks]
    results, warnings = parse_sections(SYSTEM_PROMPT, sections, ExtractedDisciplineList)

    items: list[dict] = []
    seen: set[str] = set()
    for label, parsed in results:
        for d in parsed.disciplines:
            name = re.sub(r"\s+", " ", d.name).strip()
            if not name or name.lower() == "total":
                continue
            code = d.code.strip().upper() if d.code else None
            key = code or name.lower()
            if key in seen:
                warnings.append(f"Disciplina duplicada na extração: '{name}' ({code}) — mantida a primeira")
                continue
            seen.add(key)
            if d.period is not None and not 1 <= d.period <= 12:
                warnings.append(f"'{name}': período {d.period} fora do esperado — mantido null")
                d.period = None
            items.append(
                {
                    "name": name,
                    "code": code,
                    "period": d.period,
                    "workload": d.workload,
                    "prerequisites_text": d.prerequisites_text,
                }
            )

    if not items:
        warnings.append("Nenhuma disciplina extraída do PDF.")
    return items, warnings


def _key(code: str | None, name: str) -> str:
    return code or name.lower()


def diff(db: Session, items: list[dict], course_id: int) -> tuple[dict, list[str]]:
    """Compara a matriz extraída com as disciplinas atuais do curso."""
    existing = db.query(Discipline).filter_by(course_id=course_id).all()
    existing_by_key = {_key(d.code, d.name): d for d in existing}
    new_keys = {_key(i["code"], i["name"]) for i in items}

    added, updated, unchanged = 0, 0, 0
    for item in items:
        current = existing_by_key.get(_key(item["code"], item["name"]))
        if current is None:
            added += 1
        elif (
            current.name != item["name"]
            or current.period != item["period"]
            or current.workload != item["workload"]
            or (current.prerequisites_text or None) != (item["prerequisites_text"] or None)
        ):
            updated += 1
        else:
            unchanged += 1

    warnings: list[str] = []
    orphans = [d for k, d in existing_by_key.items() if k not in new_keys]
    removable = 0
    for d in orphans:
        has_assignments = (
            db.query(ProfessorDiscipline).filter_by(discipline_id=d.id).first() is not None
        )
        if has_assignments:
            warnings.append(
                f"'{d.name}' ({d.code or 'sem sigla'}) não está no novo PPC, mas tem professores "
                "vinculados — será mantida."
            )
        else:
            removable += 1

    stats = {
        "extracted": len(items),
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "removed": removable,
        "kept_with_assignments": len(orphans) - removable,
    }
    return stats, warnings


def apply(db: Session, items: list[dict], course_id: int) -> dict:
    """Upsert por (course_id, code); remove órfãs sem vínculo com professores."""
    if not items:
        raise ValueError("Nada a aplicar: payload vazio")

    existing = db.query(Discipline).filter_by(course_id=course_id).all()
    existing_by_key = {_key(d.code, d.name): d for d in existing}
    new_keys = set()

    inserted, updated = 0, 0
    for item in items:
        key = _key(item["code"], item["name"])
        new_keys.add(key)
        current = existing_by_key.get(key)
        if current is None:
            db.add(Discipline(course_id=course_id, **item))
            inserted += 1
        else:
            changed = False
            for field in ("name", "code", "period", "workload", "prerequisites_text"):
                if getattr(current, field) != item[field]:
                    setattr(current, field, item[field])
                    changed = True
            if changed:
                updated += 1

    deleted = 0
    for key, discipline in existing_by_key.items():
        if key in new_keys:
            continue
        has_assignments = (
            db.query(ProfessorDiscipline).filter_by(discipline_id=discipline.id).first() is not None
        )
        if not has_assignments:
            db.delete(discipline)
            deleted += 1

    return {"inserted": inserted, "updated": updated, "deleted": deleted}
