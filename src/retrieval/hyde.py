"""
HyDE — Hypothetical Document Embeddings

Antes da busca densa, gera uma resposta hipotética no estilo normativo
da UNIVASF para a pergunta do usuário. O embedding desse documento
hipotético é usado na busca vetorial no lugar do embedding da pergunta bruta.

Por quê funciona:
    Perguntas em linguagem natural ("Como faço trancamento?") e documentos
    em linguagem normativa ("Art. 45. O discente poderá requerer...") têm
    embeddings distantes no espaço vetorial. O documento hipotético é gerado
    no mesmo "dialeto" dos chunks reais, reduzindo esse gap semântico.

Referência: Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" (HyDE).
"""

from __future__ import annotations

import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

_HYDE_SYSTEM_PROMPT = """\
Você é um especialista em normas e regulamentos da UNIVASF.
Dado uma pergunta, redija um trecho de resposta no estilo de um artigo \
de resolução ou regimento universitário — linguagem normativa, direta, \
com referências implícitas a artigos e parágrafos.
Não use a palavra "Hipotético". Escreva como se fosse um trecho real extraído \
de um documento oficial. Máximo 3 parágrafos curtos.\
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_hypothetical_document(query: str) -> str:
    """
    Gera um documento hipotético no estilo normativo para a query fornecida.

    Usa gpt-4o-mini com temperatura baixa para gerar um texto curto
    que imita a linguagem dos documentos normativos indexados.

    Args:
        query: Pergunta do usuário em linguagem natural.

    Returns:
        Texto hipotético no estilo de artigo normativo.
        Em caso de falha, retorna a query original como fallback.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Pergunta: {query}"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        hypothetical = response.choices[0].message.content or query
        logger.info(f"HyDE gerado ({len(hypothetical)} chars) para: '{query[:60]}...'")
        return hypothetical
    except Exception as e:
        logger.warning(f"HyDE falhou, usando query original como fallback: {e}")
        return query
