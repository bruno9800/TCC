"""
Documents Router — Download de PDFs Normativos

Expõe os PDFs originais para download, resolvendo o nome do documento
(campo `source` retornado pelo /chat/) para o arquivo físico.

O mapa stem→path é construído uma vez ao importar o módulo, varrendo
recursivamente o diretório de documentos.
"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Índice de documentos ───────────────────────────────────────────────────────
# Mapa: stem normalizado → Path absoluto do PDF
# Construído uma vez ao importar — tolerante a variações de case/espaço.

def _build_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for pdf in root.rglob("*.pdf"):
        key = pdf.stem.strip().lower()
        index[key] = pdf
    logger.info(f"Índice de documentos: {len(index)} PDFs encontrados em '{root}'")
    return index


_DOC_INDEX: dict[str, Path] = _build_index(DOCUMENTS_DIR)


def resolve_pdf(source: str) -> Path | None:
    """
    Resolve o nome do documento (campo `source` do SourceInfo) para o Path do PDF.

    Tenta correspondência exata (normalizada) primeiro. Se não encontrar,
    tenta correspondência parcial (source contido no stem do arquivo).
    """
    key = source.strip().lower()

    # Correspondência exata
    if key in _DOC_INDEX:
        return _DOC_INDEX[key]

    # Correspondência parcial — útil se o source foi truncado ou tem sufixo extra
    for stem, path in _DOC_INDEX.items():
        if key in stem or stem in key:
            return path

    return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/download",
    summary="Download do PDF original de um documento normativo",
    description=(
        "Retorna o arquivo PDF correspondente ao campo `source` "
        "retornado pelo endpoint /chat/. O parâmetro `source` deve ser "
        "passado como string (URL-encoded se necessário)."
    ),
    response_class=FileResponse,
)
async def download_document(source: str) -> FileResponse:
    """
    Faz download do PDF original referenciado em uma resposta do chat.

    Args:
        source: Nome do documento (campo `source` do SourceInfo).
                Exemplo: "Resolução 08_2015 - Normas_gerais_Graduação"
    """
    decoded = urllib.parse.unquote(source)
    pdf_path = resolve_pdf(decoded)

    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Documento '{decoded}' não encontrado. "
                   f"Verifique se o campo `source` foi copiado corretamente da resposta do /chat/.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.get(
    "/list",
    summary="Lista todos os documentos disponíveis para download",
    response_model=list[dict],
)
async def list_documents() -> list[dict]:
    """
    Retorna a lista de todos os PDFs indexados com nome e categoria inferida.
    """
    docs = []
    for pdf in sorted(DOCUMENTS_DIR.rglob("*.pdf")):
        relative = pdf.relative_to(DOCUMENTS_DIR)
        parts = relative.parts
        category = parts[0] if len(parts) > 1 else "raiz"
        docs.append({
            "source": pdf.stem,
            "filename": pdf.name,
            "category": category,
            "download_url": f"/documents/download?source={urllib.parse.quote(pdf.stem)}",
        })
    return docs
