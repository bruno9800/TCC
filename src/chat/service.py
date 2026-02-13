"""
Chat Service — Agente Leve com Pipeline RAG

Implementa um agente leve que:
1. Recebe a mensagem + histórico de conversa
2. Usa o LLM para decidir se precisa buscar nos documentos ou não
3. Se precisa: executa busca → rerank → monta contexto → gera resposta
4. Se não precisa: responde diretamente (cumprimentos, follow-ups, etc.)
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import rerank
from src.generation.generator import build_context
from src.chat.schemas import (
    ChatMessage,
    ChatResponse,
    SourceInfo,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────────

_search_engine: HybridSearchEngine | None = None
_openai_client: OpenAI | None = None


def get_search_engine() -> HybridSearchEngine:
    """Retorna instância singleton do motor de busca."""
    global _search_engine
    if _search_engine is None:
        logger.info("Inicializando HybridSearchEngine...")
        _search_engine = HybridSearchEngine()
        logger.info(f"HybridSearchEngine carregado com {len(_search_engine.chunks)} chunks.")
    return _search_engine


def get_openai_client() -> OpenAI:
    """Retorna instância singleton do cliente OpenAI."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Você é o Assistente Acadêmico da UNIVASF (Universidade Federal do Vale do São Francisco).

## Suas capacidades:
- Você tem acesso a uma ferramenta de busca nos documentos normativos oficiais da UNIVASF \
(estatutos, regimentos, resoluções).
- Quando o usuário pergunta sobre normas, regras, prazos, direitos ou procedimentos \
acadêmicos, você DEVE buscar nos documentos.

## Como decidir se precisa buscar:
- BUSCAR: perguntas sobre normas, regras, trancamento, matrícula, estágio, regulamentos, \
prazos, direitos, resoluções, artigos, etc.
- NÃO BUSCAR: cumprimentos (oi, bom dia), agradecimentos, perguntas sobre você mesmo, \
esclarecimentos sobre algo que você já respondeu na conversa.

## Formato de decisão:
Quando receber uma mensagem, responda APENAS com um JSON:
{"needs_search": true, "search_query": "consulta otimizada para busca semântica"}
ou
{"needs_search": false, "direct_response": "sua resposta direta aqui"}

IMPORTANTE: Responda SOMENTE o JSON, sem texto adicional.\
"""

ANSWER_PROMPT = """\
Você é o assistente normativo da UNIVASF. Responda dúvidas \
sobre normas, regulamentos, estatutos e resoluções da universidade.

## Regras:

1. Responda APENAS com base no contexto fornecido. Se não encontrar, diga claramente.
2. NUNCA invente normas ou artigos.
3. Cite SEMPRE a fonte: "Segundo o Art. X da [Norma]..." ou "(Art. X, [Norma])".
4. Se múltiplas normas tratam do assunto, cite todas.
5. Use linguagem clara e acessível.

## Formato:

- Seja DIRETO e CONCISO. Máximo 2-3 parágrafos curtos.
- Vá direto ao ponto: o estudante quer a resposta rápida, não um tratado.
- Evite repetições e rodeios. Não repita a pergunta.
- Cite os artigos inline, NÃO liste fontes separadamente ao final.\
"""


# ── Agente ─────────────────────────────────────────────────────────────────────


def _build_history_messages(history: list[ChatMessage]) -> list[dict]:
    """Converte o histórico de ChatMessage para o formato da API OpenAI."""
    return [{"role": msg.role, "content": msg.content} for msg in history]


def run_chat(
    message: str,
    history: list[ChatMessage],
    top_k: int = 5,
    filter_revoked: bool = True,
    model: str = LLM_MODEL,
) -> ChatResponse:
    """
    Executa o agente de chat.

    1. Envia mensagem + histórico para o LLM decidir se precisa buscar
    2. Se precisa: executa RAG pipeline e gera resposta com contexto
    3. Se não: retorna resposta direta do LLM

    Args:
        message: Pergunta do usuário.
        history: Histórico de mensagens anteriores.
        top_k: Documentos finais pós-reranking.
        filter_revoked: Se True, exclui documentos revogados.
        model: Modelo LLM a usar.

    Returns:
        ChatResponse com a resposta, fontes e métricas.
    """
    client = get_openai_client()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # ── Passo 1: LLM decide se precisa buscar ──────────────────────────────
    decision_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_build_history_messages(history),
        {"role": "user", "content": message},
    ]

    decision_response = client.chat.completions.create(
        model=model,
        messages=decision_messages,
        temperature=0.0,
        max_tokens=256,
    )

    decision_text = decision_response.choices[0].message.content or ""
    usage = decision_response.usage
    if usage:
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

    logger.info(f"Decisão do agente: {decision_text[:200]}")

    # ── Parse da decisão ───────────────────────────────────────────────────
    needs_search = True
    search_query = message
    direct_response = None

    try:
        # Tenta extrair JSON da resposta (pode ter markdown wrapping)
        clean = decision_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        decision = json.loads(clean)
        needs_search = decision.get("needs_search", True)
        if needs_search:
            search_query = decision.get("search_query", message)
        else:
            direct_response = decision.get("direct_response", "")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Falha ao parsear decisão do agente: {e}. Buscando por segurança.")
        needs_search = True
        search_query = message

    # ── Passo 2A: Resposta direta (sem busca) ──────────────────────────────
    if not needs_search and direct_response:
        logger.info("Agente respondeu diretamente (sem busca).")
        return ChatResponse(
            answer=direct_response,
            sources=[],
            model=model,
            tokens=TokenUsage(prompt=total_prompt_tokens, completion=total_completion_tokens),
            used_search=False,
        )

    # ── Passo 2B: Pipeline RAG (com busca) ─────────────────────────────────
    logger.info(f"Agente decidiu buscar: '{search_query}'")

    engine = get_search_engine()

    # Busca HNSW
    candidates = engine.search_hybrid(
        query=search_query,
        filter_revoked=filter_revoked,
    )

    # Reranking
    top_results = rerank(search_query, candidates, top_k=top_k)

    if not top_results:
        return ChatResponse(
            answer="Não encontrei informações relevantes nos documentos normativos da UNIVASF "
                   "para essa pergunta. Poderia reformular ou ser mais específico?",
            sources=[],
            model=model,
            tokens=TokenUsage(prompt=total_prompt_tokens, completion=total_completion_tokens),
            used_search=True,
        )

    # Monta contexto e gera resposta final
    context = build_context(top_results)

    answer_messages = [
        {"role": "system", "content": ANSWER_PROMPT},
        *_build_history_messages(history),
        {
            "role": "user",
            "content": (
                f"Com base nos documentos normativos fornecidos abaixo, "
                f"responda à seguinte pergunta:\n\n"
                f"**Pergunta:** {message}\n\n"
                f"**Contexto (Documentos Normativos):**\n{context}\n\n"
                f"Lembre-se: responda apenas com base nos documentos acima e cite as fontes."
            ),
        },
    ]

    answer_response = client.chat.completions.create(
        model=model,
        messages=answer_messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = answer_response.choices[0].message.content or ""
    usage = answer_response.usage
    if usage:
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

    # Monta fontes
    sources: list[SourceInfo] = []
    seen: set[str] = set()
    for result in top_results:
        source_name = result.metadata.get("source", "")
        if source_name and source_name not in seen:
            seen.add(source_name)
            sources.append(
                SourceInfo(
                    source=source_name,
                    category=result.metadata.get("category", ""),
                    article_id=result.metadata.get("article_id", ""),
                    hierarchy=result.metadata.get("hierarchy", ""),
                    score=result.score,
                    snippet=result.content[:300],
                )
            )

    return ChatResponse(
        answer=answer,
        sources=sources,
        model=model,
        tokens=TokenUsage(prompt=total_prompt_tokens, completion=total_completion_tokens),
        used_search=True,
    )
