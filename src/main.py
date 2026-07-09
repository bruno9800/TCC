"""
UNIVASF RAG API — Entrypoint FastAPI

Uso:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.admin.router import router as admin_router
from src.calendar_events.router import router as calendar_events_router
from src.chat.router import router as chat_router
from src.courses.router import router as courses_router
from src.documents.router import router as documents_router
from src.logs.router import router as logs_router
from src.professors.router import router as professors_router
from src.retrieval.hybrid_search import get_search_engine
from src.retrieval.reranker import warm_up as warm_up_reranker
from src.transport.router import router as transport_router
from src.auth import get_api_key

logger = logging.getLogger(__name__)

app = FastAPI(
    title="UNIVASF RAG API",
    description="API de Chat Inteligente para consulta de documentos normativos da UNIVASF",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir para domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rotas ──────────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/chat", tags=["Chat"], dependencies=[Depends(get_api_key)])
app.include_router(documents_router, prefix="/documents", tags=["Documents"], dependencies=[Depends(get_api_key)])
app.include_router(logs_router, prefix="/logs", tags=["Logs"], dependencies=[Depends(get_api_key)])
app.include_router(
    professors_router, prefix="/professors", tags=["Professors"], dependencies=[Depends(get_api_key)]
)
app.include_router(
    calendar_events_router,
    prefix="/academic-events",
    tags=["Academic Events"],
    dependencies=[Depends(get_api_key)],
)
app.include_router(
    courses_router, prefix="/courses", tags=["Courses"], dependencies=[Depends(get_api_key)]
)
app.include_router(
    transport_router,
    prefix="/transport-routes",
    tags=["Transport"],
    dependencies=[Depends(get_api_key)],
)
# Autenticação própria (JWT de AdminUser), deliberadamente separada da x-api-key pública.
app.include_router(admin_router, prefix="/admin", tags=["Admin"])



@app.on_event("startup")
async def warm_up_models() -> None:
    """
    Pré-carrega o reranker e o motor de busca híbrida na inicialização do
    processo, não na primeira pergunta de um usuário real — sem isso, a
    primeira resposta do chat paga o custo de carregar o Cross-Encoder
    (segundos com os pesos em cache; minutos numa máquina nova, ver o volume
    `hf_cache` no docker-compose.yml).
    """
    logger.info("Pré-carregando modelos (reranker + motor de busca)...")
    warm_up_reranker()
    get_search_engine()
    logger.info("Modelos pré-carregados — API pronta para servir requisições.")


@app.get("/health", tags=["Infra"])
async def health_check():
    """Verifica se a API está online."""
    return {"status": "ok"}
