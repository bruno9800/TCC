"""
Documents Router — Listagem, Busca Semântica e Download de PDFs Normativos

Endpoints:
  GET /documents/list          — lista todos os PDFs
  GET /documents/search?q=...  — busca semântica sobre os documentos
  GET /documents/download      — download do PDF original
"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.config import DOCUMENTS_DIR
from src.indexing.vector_store import generate_embeddings, get_or_create_collection

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
    Retorna a lista de todos os PDFs com nome, categoria e download_url.
    Use quando não há query — tela inicial antes do usuário digitar.
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


@router.get(
    "/search",
    summary="Busca semântica sobre os documentos normativos",
    response_model=list[dict],
)
async def search_documents(
    q: str = Query(..., min_length=2, description="Termo ou frase de busca"),
    limit: int = Query(default=10, ge=1, le=48, description="Número máximo de documentos retornados"),
    filter_revoked: bool = Query(default=True, description="Filtrar documentos revogados"),
) -> list[dict]:
    """
    Busca semântica em tempo real sobre os documentos normativos.

    Gera o embedding da query, consulta o ChromaDB e agrupa os chunks
    por documento-fonte, retornando os mais relevantes com score e snippet.

    Projetado para uso em campo de busca com debounce no frontend:
    chamado enquanto o usuário digita, sem HyDE (latência mínima).
    """
    try:
        query_embedding = generate_embeddings([q])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embedding: {e}")

    where_filter = {"status": "vigente"} if filter_revoked else None

    try:
        collection = get_or_create_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca vetorial: {e}")

    # Agrupa por documento-fonte — mantém apenas o chunk com melhor score por doc
    best_per_doc: dict[str, dict] = {}

    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for doc_text, meta, dist in zip(docs_list, metas_list, dists_list):
        source = meta.get("source", "")
        if not source:
            continue

        score = round(1.0 - dist, 4)

        if source not in best_per_doc or score > best_per_doc[source]["score"]:
            best_per_doc[source] = {
                "source": source,
                "filename": source + ".pdf",
                "category": meta.get("category", ""),
                "score": score,
                "snippet": doc_text[:300],
                "article_id": meta.get("article_id", ""),
                "download_url": f"/documents/download?source={urllib.parse.quote(source)}",
            }

    # Ordena por score e limita ao número pedido
    ranked = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:limit]
