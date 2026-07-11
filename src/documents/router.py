"""
Documents Router — Listagem, Busca Semântica e Download de PDFs Normativos

Endpoints:
  GET /documents/list          — lista todos os PDFs
  GET /documents/search?q=...  — busca semântica sobre os documentos
  GET /documents/download      — download do PDF original
"""

from __future__ import annotations

import logging
import unicodedata
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.db.models import Document
from src.db.session import get_db
from src.indexing.vector_store import generate_embeddings, get_or_create_collection

logger = logging.getLogger(__name__)

router = APIRouter()


def find_pdf_on_disk(storage_path: str) -> Path | None:
    """
    Resolve o arquivo em disco tolerando divergência de normalização Unicode
    no nome (NFC vs NFD).

    Os registros legados foram backfilled no macOS, que apresenta nomes de
    arquivo em NFD (decomposto: "a" + acento combinante); o git versiona os
    mesmos nomes em NFC (precomposto: "á" único) e o Linux é byte-exato — então
    no container Linux o `storage_path` NFD do banco não bate com o arquivo NFC
    em disco (checkout do git). No macOS isso passava despercebido porque o
    filesystem faz lookup insensível à forma. Tenta o caminho literal primeiro
    (uploads via admin batem exato), depois NFC e NFD.
    """
    for form in (None, "NFC", "NFD"):
        normalized = storage_path if form is None else unicodedata.normalize(form, storage_path)
        candidate = PROJECT_ROOT / normalized
        if candidate.exists():
            return candidate
    return None


def resolve_document(source: str, db: Session) -> Document | None:
    """
    Resolve o nome do documento (campo `source` do SourceInfo) para o registro
    no banco. `source` é o mesmo valor usado como `metadata.source` nos
    chunks — sempre igual a `Document.title` (nome do arquivo sem extensão),
    então a correspondência é exata (case-insensitive), sem heurística de
    filesystem: `Document.storage_path` já cobre tanto os documentos legados
    (`regimentos_estatutos_resolucoes/`) quanto os enviados via /admin
    (`data/raw/`).
    """
    return db.query(Document).filter(Document.title.ilike(source.strip())).first()


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
async def download_document(source: str, db: Session = Depends(get_db)) -> FileResponse:
    """
    Faz download do PDF original referenciado em uma resposta do chat.

    Args:
        source: Nome do documento (campo `source` do SourceInfo).
                Exemplo: "Resolução 08_2015 - Normas_gerais_Graduação"
    """
    decoded = urllib.parse.unquote(source)
    document = resolve_document(decoded, db)

    if document is None or not document.storage_path:
        raise HTTPException(
            status_code=404,
            detail=f"Documento '{decoded}' não encontrado. "
                   f"Verifique se o campo `source` foi copiado corretamente da resposta do /chat/.",
        )

    pdf_path = find_pdf_on_disk(document.storage_path)
    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo de '{decoded}' está registrado mas não foi encontrado em disco.",
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
async def list_documents(db: Session = Depends(get_db)) -> list[dict]:
    """
    Retorna a lista de todos os documentos vigentes/indexados, com nome,
    categoria e download_url. Use quando não há query — tela inicial antes
    do usuário digitar.

    Consulta o banco (Document) em vez do filesystem — cobre tanto os 48
    documentos legados quanto qualquer documento enviado via /admin.
    """
    documents = (
        db.query(Document)
        .filter(Document.status == "indexed")
        .order_by(Document.title)
        .all()
    )
    return [
        {
            "source": doc.title,
            "filename": doc.filename,
            "category": doc.category or "",
            "download_url": f"/documents/download?source={urllib.parse.quote(doc.title)}",
        }
        for doc in documents
    ]


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
