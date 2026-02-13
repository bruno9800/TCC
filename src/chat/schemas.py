"""
Schemas Pydantic — Chat API

Define os modelos de entrada e saída para o endpoint de chat.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Mensagem individual do histórico de conversa."""

    role: str = Field(..., description="Papel: 'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")


class ChatRequest(BaseModel):
    """Payload de entrada para o endpoint POST /chat."""

    message: str = Field(..., description="Pergunta do usuário")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Histórico de conversa (mensagens anteriores)",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Documentos finais pós-reranking")
    filter_revoked: bool = Field(default=True, description="Filtrar documentos revogados")


class SourceInfo(BaseModel):
    """Informações sobre uma fonte usada na resposta."""

    source: str
    category: str = ""
    article_id: str = ""
    hierarchy: str = ""
    score: float = 0.0
    snippet: str = ""


class TokenUsage(BaseModel):
    """Tokens consumidos pela geração."""

    prompt: int = 0
    completion: int = 0


class ChatResponse(BaseModel):
    """Payload de saída do endpoint POST /chat."""

    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    model: str = ""
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    used_search: bool = Field(
        default=False,
        description="Indica se o agente precisou buscar nos documentos",
    )
