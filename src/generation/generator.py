from __future__ import annotations

"""
Geração de Respostas com LLM

Constrói o prompt com persona jurídica universitária, injeta o contexto
recuperado, e gera respostas com citações obrigatórias das fontes normativas.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.retrieval.hybrid_search import SearchResult

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um assistente jurídico universitário oficial da UNIVASF \
(Universidade Federal do Vale do São Francisco). Sua função é responder perguntas \
sobre normas, regulamentos, estatutos e resoluções da universidade.

## Regras Obrigatórias:

1. **Responda APENAS com base no contexto fornecido.** Se a informação não estiver \
no contexto, declare explicitamente: "Não encontrei essa informação nos documentos \
normativos disponíveis."

2. **NUNCA invente normas, artigos ou resoluções.** Não alucine informações jurídicas.

3. **Cite SEMPRE a fonte de cada afirmação.** Use o formato: \
"Segundo o Art. X da [Nome da Norma]..." ou "(Art. X, [Nome da Norma])".

4. **Quando múltiplas normas tratam do mesmo assunto**, apresente todas as fontes \
relevantes e indique se há conflito ou complementaridade entre elas.

5. **Use linguagem clara e acessível**, mas mantenha a precisão jurídica.

6. **Estruture a resposta** com parágrafos organizados quando a resposta for longa.

7. **Ao final da resposta**, liste as fontes consultadas em formato de referência."""

# ── Funções ────────────────────────────────────────────────────────────────────

_openai_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Retorna instância singleton do cliente OpenAI."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def build_context(results: list[SearchResult]) -> str:
    """
    Formata os documentos recuperados como contexto para o LLM.

    Cada documento é precedido por seus metadados para facilitar a citação.
    """
    context_parts: list[str] = []

    for i, result in enumerate(results, 1):
        meta = result.metadata
        source = meta.get("source", "Documento desconhecido")
        article = meta.get("article_id", "")
        hierarchy = meta.get("hierarchy", "")
        category = meta.get("category", "")

        header = f"[Documento {i}]"
        header += f"\nFonte: {source}"
        if category:
            header += f" ({category})"
        if article:
            header += f"\nDispositivo: {article}"
        if hierarchy:
            header += f"\nHierarquia: {hierarchy}"
        header += f"\n{'─' * 40}"

        context_parts.append(f"{header}\n{result.content}")

    return "\n\n" + "\n\n".join(context_parts) + "\n\n"


@dataclass
class GenerationResult:
    """Resultado da geração."""

    answer: str
    sources: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


def generate_answer(
    query: str,
    context_results: list[SearchResult],
    model: str = LLM_MODEL,
    temperature: float = 0.1,
) -> GenerationResult:
    """
    Gera uma resposta fundamentada usando o LLM.

    Args:
        query: Pergunta do usuário.
        context_results: Documentos recuperados e rerankeados.
        model: Modelo LLM a usar.
        temperature: Temperatura de geração (baixa = mais preciso).

    Returns:
        GenerationResult com a resposta e metadados.
    """
    client = get_client()
    context = build_context(context_results)

    user_message = f"""Com base nos documentos normativos fornecidos abaixo, \
responda à seguinte pergunta:

**Pergunta:** {query}

**Contexto (Documentos Normativos):**
{context}

Lembre-se: responda apenas com base nos documentos acima e cite as fontes."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage

    # Extrai lista de fontes únicas
    sources = []
    seen_sources: set[str] = set()
    for result in context_results:
        source_name = result.metadata.get("source", "")
        if source_name and source_name not in seen_sources:
            seen_sources.add(source_name)
            sources.append({
                "source": source_name,
                "category": result.metadata.get("category", ""),
                "article_id": result.metadata.get("article_id", ""),
            })

    return GenerationResult(
        answer=answer,
        sources=sources,
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
