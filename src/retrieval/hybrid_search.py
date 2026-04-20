from __future__ import annotations

"""
Motor de Busca — HNSW Dense Search via ChromaDB

Utiliza o algoritmo HNSW (Hierarchical Navigable Small World) implementado
pelo ChromaDB para busca vetorial densa eficiente.
Opcionalmente suporta busca híbrida com BM25 (desativada por padrão).
"""

import logging
from dataclasses import dataclass

from src.config import INITIAL_TOP_K
from src.chunking.legal_chunker import LegalChunk, load_all_chunks
from src.indexing.vector_store import query_dense

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de busca com score."""

    content: str
    metadata: dict
    score: float
    source: str  # "dense" ou "hybrid"


class HybridSearchEngine:
    """
    Motor de busca vetorial usando HNSW (via ChromaDB).

    Por padrão utiliza apenas busca densa (HNSW), que é significativamente
    mais rápida que a combinação com BM25. O modo híbrido com BM25 pode
    ser ativado opcionalmente via parâmetro.
    """

    def __init__(self, chunks: list[LegalChunk] | None = None, use_bm25: bool = True):
        """
        Inicializa o motor de busca.

        Args:
            chunks: Lista de chunks (carrega do disco se None).
            use_bm25: Se True, constrói índice BM25 para busca híbrida.
        """
        if chunks is None:
            chunks = load_all_chunks()

        self.chunks = chunks
        self.bm25 = None
        self.corpus_tokens = None

        if use_bm25:
            self._build_bm25_index()

    def _build_bm25_index(self):
        """Constrói o índice BM25 (opcional, sob demanda)."""
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = [
            chunk.content.lower().split() for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Índice BM25 construído com {len(self.chunks)} documentos")

    def search_dense(
        self,
        query: str,
        top_k: int = INITIAL_TOP_K,
        filter_revoked: bool = True,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """
        Busca densa via HNSW (ChromaDB).

        O ChromaDB utiliza internamente o algoritmo HNSW (Hierarchical
        Navigable Small World) para busca aproximada de vizinhos mais
        próximos, garantindo busca sub-linear eficiente.

        Args:
            query: Consulta do usuário.
            top_k: Número de resultados.
            filter_revoked: Se True, exclui documentos revogados.
            query_embedding: Embedding pré-computado (ex: gerado pelo HyDE).
                             Se fornecido, o embedding de `query` não é gerado.

        Returns:
            Lista de SearchResult ordenada por similaridade cosseno.
        """
        where_filter = {"status": "vigente"} if filter_revoked else None

        results = query_dense(
            query=query,
            top_k=top_k,
            where_filter=where_filter,
            query_embedding=query_embedding,
        )

        search_results: list[SearchResult] = []

        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 1.0
                # Converte distância cosseno em score de similaridade
                similarity = 1.0 - dist

                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta,
                        score=similarity,
                        source="dense",
                    )
                )

        logger.info(f"Busca HNSW: {len(search_results)} resultados para top_k={top_k}")
        return search_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = INITIAL_TOP_K,
        filter_revoked: bool = True,
        hyde_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """
        Busca híbrida: HNSW dense + BM25 esparso, fundidos via RRF.

        Se hyde_embedding for fornecido, a busca densa usa o embedding do
        documento hipotético (HyDE) em vez do embedding da query original.
        O BM25 sempre usa os tokens da query original.

        Args:
            query: Consulta do usuário (usada pelo BM25 e como fallback do dense).
            top_k: Número de resultados.
            filter_revoked: Se True, exclui documentos revogados.
            hyde_embedding: Embedding pré-computado pelo HyDE (opcional).

        Returns:
            Lista de SearchResult ordenada por relevância.
        """
        dense_results = self.search_dense(
            query,
            top_k=top_k,
            filter_revoked=filter_revoked,
            query_embedding=hyde_embedding,
        )

        # Se BM25 não está ativado, retorna apenas dense (HNSW)
        if self.bm25 is None:
            return dense_results

        # ── Modo Híbrido (RRF) ─────────────────────────────────────────────
        from src.config import BM25_WEIGHT, DENSE_WEIGHT

        k = 60  # constante RRF

        # Busca esparsa
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        sparse_results: list[SearchResult] = []
        for idx, score in scored_indices:
            if score > 0:
                chunk = self.chunks[idx]
                meta = {
                    "source": chunk.metadata.source,
                    "category": chunk.metadata.category,
                    "status": chunk.metadata.status,
                    "article_id": chunk.metadata.article_id,
                    "hierarchy": " > ".join(chunk.metadata.hierarchy),
                }
                if filter_revoked and meta["status"] == "revogado":
                    continue
                sparse_results.append(
                    SearchResult(content=chunk.content, metadata=meta,
                                 score=float(score), source="sparse")
                )

        # Fusão RRF
        content_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results):
            key = result.content[:200]
            rrf = DENSE_WEIGHT * (1.0 / (k + rank + 1))
            if key not in content_map:
                content_map[key] = {"result": result, "rrf_score": rrf}
            else:
                content_map[key]["rrf_score"] += rrf

        for rank, result in enumerate(sparse_results):
            key = result.content[:200]
            rrf = BM25_WEIGHT * (1.0 / (k + rank + 1))
            if key not in content_map:
                content_map[key] = {"result": result, "rrf_score": rrf}
            else:
                content_map[key]["rrf_score"] += rrf

        sorted_results = sorted(
            content_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]

        return [
            SearchResult(
                content=item["result"].content,
                metadata=item["result"].metadata,
                score=item["rrf_score"],
                source="hybrid",
            )
            for item in sorted_results
        ]
