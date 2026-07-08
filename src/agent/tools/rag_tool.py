"""
RagTool — busca em documentos normativos oficiais da UNIVASF (RAG)

Wrap fino sobre o pipeline de recuperação já existente (HyDE + busca híbrida
+ reranking) — nenhuma lógica de retrieval é reimplementada aqui, só
reorganizada para ser chamável pelo AgentOrchestrator (Fase 4).

Contrato de tool (ver src/agent/orchestrator.py): execute(arguments, context)
retorna {"summary": str, "sources": list[dict]} — "summary" volta pro LLM
como conteúdo da mensagem `role: "tool"`; "sources" alimenta o ChatResponse.
"""

from __future__ import annotations

from src.config import RERANK_TOP_K
from src.generation.generator import build_context
from src.indexing.vector_store import generate_embeddings
from src.retrieval.hybrid_search import get_search_engine
from src.retrieval.hyde import generate_hypothetical_document
from src.retrieval.reranker import rerank

NAME = "search_normative_documents"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Busca em documentos normativos oficiais da UNIVASF (estatutos, regimentos, "
            "resoluções) para responder perguntas sobre normas, regras, prazos, direitos "
            "ou procedimentos acadêmicos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta otimizada para busca semântica nos documentos normativos.",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(arguments: dict, context: dict) -> dict:
    query = arguments["query"]
    filter_revoked = context.get("filter_revoked", True)
    top_k = context.get("top_k", RERANK_TOP_K)

    hypothetical_doc = generate_hypothetical_document(query)
    hyde_embedding = generate_embeddings([hypothetical_doc])[0]

    engine = get_search_engine()
    candidates = engine.search_hybrid(
        query=query,
        filter_revoked=filter_revoked,
        hyde_embedding=hyde_embedding,
        course_id=context.get("course_id"),
    )
    top_results = rerank(query, candidates, top_k=top_k)

    if not top_results:
        return {
            "summary": "Nenhum documento normativo relevante foi encontrado para essa consulta.",
            "sources": [],
        }

    summary = build_context(top_results)

    sources: list[dict] = []
    seen: set[str] = set()
    for result in top_results:
        source_name = result.metadata.get("source", "")
        if source_name and source_name not in seen:
            seen.add(source_name)
            sources.append(
                {
                    "origin": "rag",
                    "source": source_name,
                    "category": result.metadata.get("category", ""),
                    "article_id": result.metadata.get("article_id", ""),
                    "hierarchy": result.metadata.get("hierarchy", ""),
                    "score": result.score,
                    "snippet": result.content[:300],
                }
            )

    return {"summary": summary, "sources": sources}
