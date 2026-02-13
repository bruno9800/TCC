"""
Chunking Semântico-Hierárquico para Documentos Legais

Implementa segmentação baseada na estrutura de dispositivos legais
(Lei Complementar 95/1998): Artigos, Parágrafos, Incisos, Alíneas.

Cada chunk corresponde a um Artigo completo (caput + parágrafos + incisos).
Artigos excessivamente longos são divididos com herança de contexto do caput.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import tiktoken

from src.config import CHUNKS_DIR, MAX_CHUNK_TOKENS, OVERLAP_TOKENS

logger = logging.getLogger(__name__)

# ── Regex Patterns para Estrutura Legal ────────────────────────────────────────

# Detecta início de artigo: "Art. 1º", "Art. 12.", "Art. 100 -", "Art. 1° -"
ARTICLE_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*{0,2})"       # possíveis headers markdown e bold
    r"(Art\.?\s*\d+[º°]?\.?\s*[-–]?\s*)", # "Art. 15º -"
    re.MULTILINE,
)

# Detecta títulos, capítulos, seções (hierarquia)
TITLE_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(T[ÍI]TULO\s+[IVXLCDM]+)"
    r"(?:\s*[-–:]\s*(.+))?",
    re.MULTILINE | re.IGNORECASE,
)

CHAPTER_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(CAP[ÍI]TULO\s+[IVXLCDM]+)"
    r"(?:\s*[-–:]\s*(.+))?",
    re.MULTILINE | re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(SE[ÇC][ÃA]O\s+[IVXLCDM]+)"
    r"(?:\s*[-–:]\s*(.+))?",
    re.MULTILINE | re.IGNORECASE,
)

# Tokenizer para contagem de tokens
_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkMetadata:
    """Metadados de um chunk."""

    hierarchy: list[str] = field(default_factory=list)
    source: str = ""
    category: str = ""
    status: str = "vigente"
    article_id: str = ""
    chunk_index: int = 0
    is_child_chunk: bool = False
    parent_article: str = ""


@dataclass
class LegalChunk:
    """Um chunk de documento legal com conteúdo e metadados."""

    content: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": asdict(self.metadata),
        }

    def token_count(self) -> int:
        return len(_encoder.encode(self.content))


def count_tokens(text: str) -> int:
    """Conta o número de tokens em um texto."""
    return len(_encoder.encode(text))


def extract_hierarchy(text: str, up_to_pos: int) -> list[str]:
    """
    Extrai a hierarquia (Título > Capítulo > Seção) vigente em uma posição
    do texto.

    Args:
        text: O texto completo do documento Markdown.
        up_to_pos: Posição no texto até onde buscar.

    Returns:
        Lista de strings representando a hierarquia, ex:
        ["Título I - Da Universidade", "Capítulo II - Do Ensino"]
    """
    hierarchy: list[str] = []
    search_text = text[:up_to_pos]

    # Encontra o último título antes da posição
    titles = list(TITLE_PATTERN.finditer(search_text))
    if titles:
        match = titles[-1]
        title = match.group(1).strip()
        subtitle = match.group(2).strip() if match.group(2) else ""
        hierarchy.append(f"{title} - {subtitle}" if subtitle else title)

    # Encontra o último capítulo
    chapters = list(CHAPTER_PATTERN.finditer(search_text))
    if chapters:
        match = chapters[-1]
        chapter = match.group(1).strip()
        subtitle = match.group(2).strip() if match.group(2) else ""
        hierarchy.append(f"{chapter} - {subtitle}" if subtitle else chapter)

    # Encontra a última seção
    sections = list(SECTION_PATTERN.finditer(search_text))
    if sections:
        match = sections[-1]
        section = match.group(1).strip()
        subtitle = match.group(2).strip() if match.group(2) else ""
        hierarchy.append(f"{section} - {subtitle}" if subtitle else section)

    return hierarchy


def extract_article_id(text: str) -> str:
    """
    Extrai o identificador do artigo (ex: 'Art. 15') de um texto de chunk.
    """
    match = re.match(
        r"(?:\*{0,2})(Art\.?\s*\d+[º°]?)",
        text.strip(),
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def split_into_articles(markdown_text: str) -> list[tuple[int, str]]:
    """
    Divide um documento Markdown em blocos, cada um correspondente a um Artigo.

    Texto antes do primeiro artigo (preâmbulo, títulos, etc.) é incluído
    como um bloco separado se contiver conteúdo relevante.

    Returns:
        Lista de tuplas (posição_no_texto, conteúdo_do_artigo)
    """
    matches = list(ARTICLE_PATTERN.finditer(markdown_text))

    if not matches:
        # Documento sem artigos — retorna o texto inteiro como um chunk
        return [(0, markdown_text.strip())]

    blocks: list[tuple[int, str]] = []

    # Texto antes do primeiro artigo (preâmbulo)
    preamble = markdown_text[: matches[0].start()].strip()
    if preamble and len(preamble) > 50:  # ignora preâmbulos vazios
        blocks.append((0, preamble))

    # Cada artigo vai do início do match até o próximo match
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        article_text = markdown_text[start:end].strip()
        if article_text:
            blocks.append((start, article_text))

    return blocks


def split_long_chunk(
    article_text: str,
    metadata: ChunkMetadata,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[LegalChunk]:
    """
    Divide um artigo longo em sub-chunks, mantendo o caput como prefixo
    de contexto herdado (conforme tcc_rules.md, Passo 3.3).

    Estratégia:
    1. Identifica o caput (primeira sentença/parágrafo do artigo)
    2. Divide o restante em chunks de tamanho max_tokens
    3. Cada sub-chunk recebe o caput como prefixo

    Args:
        article_text: O texto completo do artigo.
        metadata: Metadados base do chunk.
        max_tokens: Limite de tokens por chunk.

    Returns:
        Lista de LegalChunk com contexto herdado.
    """
    # Identifica o caput: tudo até o primeiro parágrafo, inciso, ou quebra dupla
    # O caput geralmente é a primeira sentença que começa com "Art. X"
    caput_end_patterns = [
        re.compile(r"\n\s*§\s*\d+", re.MULTILINE),          # parágrafo
        re.compile(r"\n\s*Parágrafo\s+único", re.MULTILINE | re.IGNORECASE),
        re.compile(r"\n\s*I\s*[-–]", re.MULTILINE),          # inciso
        re.compile(r"\n\n", re.MULTILINE),                   # quebra dupla
    ]

    caput_end = len(article_text)
    for pattern in caput_end_patterns:
        match = pattern.search(article_text)
        if match and match.start() < caput_end:
            caput_end = match.start()

    caput = article_text[:caput_end].strip()
    rest = article_text[caput_end:].strip()

    if not rest:
        # Artigo é apenas o caput, não precisa dividir
        return [
            LegalChunk(
                content=article_text,
                metadata=ChunkMetadata(
                    **{
                        **asdict(metadata),
                        "chunk_index": 0,
                        "is_child_chunk": False,
                    }
                ),
            )
        ]

    # Divide o restante em partes respeitando quebras de linha
    lines = rest.split("\n")
    chunks: list[LegalChunk] = []
    current_lines: list[str] = []
    current_tokens = count_tokens(caput)  # O caput será prefixo
    chunk_idx = 0

    for line in lines:
        line_tokens = count_tokens(line)

        if current_tokens + line_tokens > max_tokens and current_lines:
            # Cria chunk com o caput como prefixo
            chunk_content = f"{caput}\n\n[...continuação...]\n\n" + "\n".join(current_lines)
            chunks.append(
                LegalChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        hierarchy=metadata.hierarchy.copy(),
                        source=metadata.source,
                        category=metadata.category,
                        status=metadata.status,
                        article_id=metadata.article_id,
                        chunk_index=chunk_idx,
                        is_child_chunk=True if chunk_idx > 0 else False,
                        parent_article=metadata.article_id,
                    ),
                )
            )
            current_lines = [line]
            current_tokens = count_tokens(caput) + line_tokens
            chunk_idx += 1
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    # Último chunk
    if current_lines:
        combined = "\n".join(current_lines)
        if chunk_idx == 0:
            # Tudo coube no primeiro chunk — usa o texto original sem "[...continuação...]"
            chunk_content = article_text
        else:
            chunk_content = f"{caput}\n\n[...continuação...]\n\n{combined}"

        chunks.append(
            LegalChunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    hierarchy=metadata.hierarchy.copy(),
                    source=metadata.source,
                    category=metadata.category,
                    status=metadata.status,
                    article_id=metadata.article_id,
                    chunk_index=chunk_idx,
                    is_child_chunk=True if chunk_idx > 0 else False,
                    parent_article=metadata.article_id,
                ),
            )
        )

    return chunks


def chunk_document(
    markdown_text: str,
    source: str,
    category: str,
    status: str = "vigente",
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[LegalChunk]:
    """
    Aplica chunking semântico-hierárquico a um documento Markdown.

    Fluxo:
    1. Divide o documento em blocos por Artigo
    2. Para cada artigo, extrai hierarquia (Título > Capítulo > Seção)
    3. Para artigos longos, aplica split com herança de contexto
    4. Enriquece cada chunk com metadados

    Args:
        markdown_text: Texto do documento em Markdown.
        source: Nome do arquivo de origem (ex: "Resolução 08/2015").
        category: Categoria do documento (ex: "Resolução PROEN").
        status: Status de vigência ("vigente" ou "revogado").
        max_tokens: Limite máximo de tokens por chunk.

    Returns:
        Lista de LegalChunk prontos para indexação.
    """
    articles = split_into_articles(markdown_text)
    all_chunks: list[LegalChunk] = []

    for position, article_text in articles:
        # Extrai hierarquia vigente nessa posição do documento
        hierarchy = extract_hierarchy(markdown_text, position)

        # Extrai ID do artigo
        article_id = extract_article_id(article_text)

        # Cria metadados base
        base_metadata = ChunkMetadata(
            hierarchy=hierarchy,
            source=source,
            category=category,
            status=status,
            article_id=article_id,
        )

        # Verifica se precisa dividir
        tokens = count_tokens(article_text)

        if tokens > max_tokens:
            # Artigo longo — dividir com herança de contexto
            sub_chunks = split_long_chunk(article_text, base_metadata, max_tokens)
            all_chunks.extend(sub_chunks)
            logger.debug(
                f"  Artigo {article_id} dividido em {len(sub_chunks)} sub-chunks "
                f"({tokens} tokens)"
            )
        else:
            # Artigo cabe em um único chunk
            all_chunks.append(
                LegalChunk(content=article_text, metadata=base_metadata)
            )

    logger.info(f"  Documento '{source}': {len(all_chunks)} chunks gerados")
    return all_chunks


def save_chunks(chunks: list[LegalChunk], output_name: str) -> Path:
    """
    Salva os chunks em formato JSONL.

    Args:
        chunks: Lista de chunks a salvar.
        output_name: Nome base do arquivo de saída (sem extensão).

    Returns:
        Caminho do arquivo JSONL salvo.
    """
    output_path = CHUNKS_DIR / f"{output_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"  Chunks salvos em: {output_path}")
    return output_path


def load_chunks(jsonl_path: Path) -> list[LegalChunk]:
    """Carrega chunks a partir de um arquivo JSONL."""
    chunks: list[LegalChunk] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            meta = ChunkMetadata(**data["metadata"])
            chunks.append(LegalChunk(content=data["content"], metadata=meta))
    return chunks


def load_all_chunks() -> list[LegalChunk]:
    """Carrega todos os chunks de todos os arquivos JSONL no diretório."""
    all_chunks: list[LegalChunk] = []
    for jsonl_file in sorted(CHUNKS_DIR.glob("*.jsonl")):
        chunks = load_chunks(jsonl_file)
        all_chunks.extend(chunks)
    logger.info(f"Total de chunks carregados: {len(all_chunks)}")
    return all_chunks
