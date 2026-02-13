from __future__ import annotations

"""
Busca Híbrida (Dense + Sparse + RRF Fusion)

Combina busca vetorial densa (ChromaDB) com busca esparsa (BM25)
usando Reciprocal Rank Fusion para maximizar recall e precisão.
"""

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import BM25_WEIGHT, DENSE_WEIGHT, INITIAL_TOP_K
from src.chunking.legal_chunker import LegalChunk, load_all_chunks
from src.indexing.vector_store import query_dense, get_or_create_collection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de busca com score combinado."""

    content: str
    metadata: dict
    score: float
    source: str  # "dense", "sparse", ou "hybrid"


class HybridSearchEngine:
    """
    Motor de busca híbrida que combina busca densa (vetorial) e
    busca esparsa (BM25) usando Reciprocal Rank Fusion.
    """

    def __init__(self, chunks: list[LegalChunk] | None = None):
        """
        Inicializa o motor de busca.

        Args:
            chunks: Lista de chunks para indexar no BM25.
                    Se None, carrega todos os chunks do disco.
        """
        if chunks is None:
            chunks = load_all_chunks()

        self.chunks = chunks
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Constrói o índice BM25 sobre o corpus de chunks."""
        # Tokeniza de forma simples (split por espaço)
        self.corpus_tokens = [
            chunk.content.lower().split() for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Índice BM25 construído com {len(self.chunks)} documentos")

    def search_sparse(self, query: str, top_k: int = INITIAL_TOP_K) -> list[SearchResult]:
        """
        Busca esparsa usando BM25.

        Args:
            query: Consulta do usuário.
            top_k: Número de resultados.

        Returns:
            Lista de SearchResult ordenada por score BM25.
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Obtém os top_k índices com maiores scores
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[SearchResult] = []
        for idx, score in scored_indices:
            if score > 0:
                chunk = self.chunks[idx]
                results.append(
                    SearchResult(
                        content=chunk.content,
                        metadata={
                            "source": chunk.metadata.source,
                            "category": chunk.metadata.category,
                            "status": chunk.metadata.status,
                            "article_id": chunk.metadata.article_id,
                            "hierarchy": " > ".join(chunk.metadata.hierarchy),
                        },
                        score=float(score),
                        source="sparse",
                    )
                )

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = INITIAL_TOP_K,
        filter_revoked: bool = True,
    ) -> list[SearchResult]:
        """
        Busca densa usando ChromaDB.

        Args:
            query: Consulta do usuário.
            top_k: Número de resultados.
            filter_revoked: Se True, exclui documentos revogados.

        Returns:
            Lista de SearchResult ordenada por similaridade.
        """
        where_filter = {"status": "vigente"} if filter_revoked else None

        results = query_dense(
            query=query,
            top_k=top_k,
            where_filter=where_filter,
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

        return search_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = INITIAL_TOP_K,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = BM25_WEIGHT,
        filter_revoked: bool = True,
    ) -> list[SearchResult]:
        """
        Busca híbrida com Reciprocal Rank Fusion (RRF).

        Combina os rankings da busca densa e esparsa usando a fórmula:
        RRF_score = Σ (1 / (k + rank_i))
        onde k=60 (constante RRF) e rank_i é a posição no ranking i.

        Args:
            query: Consulta do usuário.
            top_k: Número de resultados finais.
            dense_weight: Peso da busca densa.
            sparse_weight: Peso da busca esparsa.
            filter_revoked: Se True, exclui documentos revogados.

        Returns:
            Lista de SearchResult com score RRF combinado.
        """
        k = 60  # constante RRF padrão

        # Executa ambas as buscas
        dense_results = self.search_dense(query, top_k=top_k, filter_revoked=filter_revoked)
        sparse_results = self.search_sparse(query, top_k=top_k)

        if filter_revoked:
            sparse_results = [
                r for r in sparse_results
                if r.metadata.get("status") != "revogado"
            ]

        # Mapeia conteúdo → melhor resultado + scores
        content_map: dict[str, dict] = {}

        # Scores da busca densa
        for rank, result in enumerate(dense_results):
            content_key = result.content[:200]  # usa prefixo como chave
            rrf_score = dense_weight * (1.0 / (k + rank + 1))
            if content_key not in content_map:
                content_map[content_key] = {
                    "result": result,
                    "rrf_score": rrf_score,
                }
            else:
                content_map[content_key]["rrf_score"] += rrf_score

        # Scores da busca esparsa
        for rank, result in enumerate(sparse_results):
            content_key = result.content[:200]
            rrf_score = sparse_weight * (1.0 / (k + rank + 1))
            if content_key not in content_map:
                content_map[content_key] = {
                    "result": result,
                    "rrf_score": rrf_score,
                }
            else:
                content_map[content_key]["rrf_score"] += rrf_score

        # Ordena por RRF score
        sorted_results = sorted(
            content_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        hybrid_results: list[SearchResult] = []
        for item in sorted_results:
            r = item["result"]
            hybrid_results.append(
                SearchResult(
                    content=r.content,
                    metadata=r.metadata,
                    score=item["rrf_score"],
                    source="hybrid",
                )
            )

        logger.info(
            f"Busca híbrida: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(hybrid_results)} resultados"
        )
        return hybrid_results
