"""
Heading Chunker — Fallback para Conteúdo sem Estrutura de Artigo Legal

Usado por legal_chunker.chunk_document() para blocos de texto (ou documentos
inteiros) sem "Art. X" — ex: o corpo narrativo de um PPC (identificação,
ementário, infraestrutura) ou um futuro Manual do Aluno. Divide por headings
Markdown; sub-divide por parágrafo se uma seção ainda for grande demais.

Mesmo contrato de saída que legal_chunker (LegalChunk/ChunkMetadata) — nada
a jusante (indexação, busca, geração) precisa saber qual chunker gerou o quê.
"""

from __future__ import annotations

import re

HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Retorna [(heading, conteúdo_da_seção), ...]. Sem headings, retorna [("", text)]."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    lead = text[: matches[0].start()].strip()
    if lead:
        sections.append(("", lead))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        sections.append((heading, text[start:end].strip()))

    return sections


def _split_by_paragraphs(text: str, max_tokens: int, count_tokens) -> list[str]:
    """Divide um texto grande em partes ~max_tokens, respeitando parágrafos (quebra dupla)."""
    paragraphs = text.split("\n\n")
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            parts.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        parts.append("\n\n".join(current))

    return parts


def split_prose_block(block_text: str, base_metadata, max_tokens: int) -> list:
    """
    Divide um bloco de prosa (sem "Art. X") em chunks por heading, com
    overflow por parágrafo quando uma seção sozinha excede max_tokens.

    Args:
        block_text: Texto do bloco (documento inteiro ou o preâmbulo de um
            documento misto — ver legal_chunker.chunk_document).
        base_metadata: ChunkMetadata base (hierarchy/source/category/status/
            kb_slug/course_id) a herdar por cada chunk gerado.
        max_tokens: Limite de tokens por chunk antes de subdividir por parágrafo.

    Returns:
        Lista de LegalChunk.
    """
    # Import local para evitar import circular (legal_chunker chama este
    # módulo; este módulo precisa dos tipos definidos em legal_chunker).
    from src.chunking.legal_chunker import ChunkMetadata, LegalChunk, count_tokens

    sections = _split_by_headings(block_text)
    chunks: list[LegalChunk] = []
    chunk_index = 0

    for heading, content in sections:
        if not content.strip():
            continue

        section_hierarchy = base_metadata.hierarchy + [heading] if heading else list(base_metadata.hierarchy)

        if count_tokens(content) > max_tokens:
            parts = _split_by_paragraphs(content, max_tokens, count_tokens)
        else:
            parts = [content]

        for part in parts:
            if not part.strip():
                continue
            chunks.append(
                LegalChunk(
                    content=part,
                    metadata=ChunkMetadata(
                        hierarchy=section_hierarchy,
                        source=base_metadata.source,
                        category=base_metadata.category,
                        status=base_metadata.status,
                        article_id="",
                        chunk_index=chunk_index,
                        is_child_chunk=chunk_index > 0,
                        kb_slug=base_metadata.kb_slug,
                        course_id=base_metadata.course_id,
                    ),
                )
            )
            chunk_index += 1

    return chunks
