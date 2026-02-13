from __future__ import annotations

"""
Cross-Encoder Reranker

Reordena os candidatos da busca híbrida usando um modelo
Cross-Encoder para capturar nuances sintáticas profundas.
Reduz top-50 → top-5 para enviar ao LLM.
"""

import logging

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K
from src.retrieval.hybrid_search import SearchResult

logger = logging.getLogger(__name__)

# Singleton do modelo de reranking
_reranker_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Retorna uma instância singleton do Cross-Encoder."""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Carregando modelo de reranking: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        logger.info("Modelo de reranking carregado.")
    return _reranker_model


def rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int = RERANK_TOP_K,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """
    Reordena os candidatos usando um Cross-Encoder.

    O Cross-Encoder processa cada par (query, document) simultaneamente,
    capturando interações profundas entre query e documento que a busca
    vetorial simples (Bi-Encoder) não consegue capturar.

    Args:
        query: A pergunta do usuário.
        candidates: Lista de SearchResult da busca híbrida.
        top_k: Número de resultados finais após reranking.
        score_threshold: Score mínimo para inclusão (opcional).

    Returns:
        Lista de SearchResult reordenada, limitada a top_k.
    """
    if not candidates:
        return []

    model = get_reranker()

    # Prepara pares (query, document) para o cross-encoder
    pairs = [(query, result.content) for result in candidates]

    # Calcula scores
    scores = model.predict(pairs)

    # Associa scores aos resultados
    scored_results: list[tuple[float, SearchResult]] = []
    for score, result in zip(scores, candidates):
        reranked = SearchResult(
            content=result.content,
            metadata=result.metadata,
            score=float(score),
            source="reranked",
        )
        scored_results.append((float(score), reranked))

    # Ordena por score decrescente
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Aplica threshold se configurado
    if score_threshold is not None:
        scored_results = [
            (s, r) for s, r in scored_results if s >= score_threshold
        ]

    # Seleciona top_k
    final = [result for _, result in scored_results[:top_k]]

    logger.info(
        f"Reranking: {len(candidates)} candidatos → {len(final)} selecionados "
        f"(scores: {scored_results[0][0]:.4f} … {scored_results[min(top_k-1, len(scored_results)-1)][0]:.4f})"
    )

    return final
