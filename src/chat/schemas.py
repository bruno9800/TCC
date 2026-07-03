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

    origin: str = Field(
        default="rag",
        description="Origem da fonte: 'rag' (documento normativo) ou 'professor' (corpo docente).",
    )
    source: str
    category: str = ""
    article_id: str = ""
    hierarchy: str = ""
    score: float = 0.0
    snippet: str = ""
    download_url: str = Field(
        default="",
        description="URL relativa para download do PDF original. "
                    "Use GET {base_url}{download_url} para baixar o arquivo. "
                    "Vazio para fontes que não são documentos (ex: origin='professor').",
    )


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
        description="Indica se o agente precisou acionar alguma ferramenta (RAG ou estruturada)",
    )
    used_tools: list[str] = Field(
        default_factory=list,
        description="Nomes das ferramentas acionadas nesta resposta (ex: ['search_normative_documents']).",
    )
