from __future__ import annotations

"""
Vector Store — ChromaDB com OpenAI Embeddings

Gerencia a indexação e persistência dos chunks usando ChromaDB
com embeddings gerados pela API da OpenAI.
"""

import logging
from dataclasses import asdict
from pathlib import Path

import chromadb
from openai import OpenAI

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
)
from src.chunking.legal_chunker import LegalChunk

logger = logging.getLogger(__name__)

# ── Cliente OpenAI ─────────────────────────────────────────────────────────────
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Retorna uma instância singleton do cliente OpenAI."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _sanitize_text(text: str) -> str:
    """Sanitiza texto para a API de embeddings — remove strings vazias e NULLs."""
    if not text:
        return " "  # API rejeita strings vazias
    # Remove caracteres nulos e espaços excessivos
    cleaned = text.replace("\x00", "").strip()
    return cleaned if cleaned else " "


def generate_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos usando a API da OpenAI.

    Args:
        texts: Lista de strings para vetorizar.
        model: Nome do modelo de embedding.

    Returns:
        Lista de vetores (cada um é uma lista de floats).
    """
    client = get_openai_client()

    # Sanitiza todos os textos
    sanitized = [_sanitize_text(t) for t in texts]

    # Batch menor para evitar payload too large em chunks longos
    batch_size = 256
    all_embeddings: list[list[float]] = []

    for i in range(0, len(sanitized), batch_size):
        batch = sanitized[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            logger.info(f"  Embeddings gerados: {i + len(batch)}/{len(sanitized)}")
        except Exception as e:
            logger.error(f"  Erro no batch {i}-{i+len(batch)}: {e}")
            # Tenta enviar um por um para identificar o problemático
            for j, text in enumerate(batch):
                try:
                    resp = client.embeddings.create(input=[text], model=model)
                    all_embeddings.append(resp.data[0].embedding)
                except Exception as inner_e:
                    logger.error(
                        f"    Chunk #{i+j} falhou ({len(text)} chars): {inner_e}"
                    )
                    # Usa embedding de placeholder
                    resp = client.embeddings.create(input=[" "], model=model)
                    all_embeddings.append(resp.data[0].embedding)

    return all_embeddings


# ── ChromaDB ───────────────────────────────────────────────────────────────────


def get_chroma_client() -> chromadb.PersistentClient:
    """Retorna um cliente ChromaDB com persistência em disco."""
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


def get_or_create_collection(
    client: chromadb.PersistentClient | None = None,
    name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Obtém ou cria a collection ChromaDB para os chunks."""
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # distância cosseno
    )
    return collection


def index_chunks(
    chunks: list[LegalChunk],
    collection: chromadb.Collection | None = None,
) -> chromadb.Collection:
    """
    Indexa os chunks no ChromaDB com embeddings OpenAI.

    Args:
        chunks: Lista de LegalChunk para indexar.
        collection: Collection do ChromaDB (cria se None).

    Returns:
        A collection do ChromaDB com os chunks indexados.
    """
    if collection is None:
        collection = get_or_create_collection()

    # Prepara os dados
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = (
            f"{chunk.metadata.source}__"
            f"{chunk.metadata.article_id or 'preamble'}__"
            f"{chunk.metadata.chunk_index}"
        ).replace(" ", "_").replace("/", "-")

        # Garantir que IDs são únicos
        if chunk_id in ids:
            chunk_id = f"{chunk_id}__{i}"

        ids.append(chunk_id)
        documents.append(chunk.content)

        # ChromaDB aceita apenas str, int, float, bool em metadados
        flat_meta = {
            "source": chunk.metadata.source,
            "category": chunk.metadata.category,
            "status": chunk.metadata.status,
            "article_id": chunk.metadata.article_id,
            "hierarchy": " > ".join(chunk.metadata.hierarchy),
            "chunk_index": chunk.metadata.chunk_index,
            "is_child_chunk": chunk.metadata.is_child_chunk,
        }
        metadatas.append(flat_meta)

    logger.info(f"Gerando embeddings para {len(documents)} chunks...")
    embeddings = generate_embeddings(documents)

    # ChromaDB aceita upsert em batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info(f"  Indexados: {min(end, len(ids))}/{len(ids)}")

    logger.info(f"Indexação completa: {collection.count()} chunks na collection")
    return collection


def query_dense(
    query: str,
    top_k: int = 50,
    where_filter: dict | None = None,
    collection: chromadb.Collection | None = None,
) -> dict:
    """
    Realiza busca densa (vetorial) no ChromaDB.

    Args:
        query: Pergunta ou consulta do usuário.
        top_k: Número de resultados a retornar.
        where_filter: Filtro de metadados (ex: {"status": "vigente"}).
        collection: Collection do ChromaDB.

    Returns:
        Dicionário com ids, documents, metadatas, distances.
    """
    if collection is None:
        collection = get_or_create_collection()

    # Gera embedding da query
    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return results
