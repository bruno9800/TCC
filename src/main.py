"""
UNIVASF RAG API — Entrypoint FastAPI

Uso:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.admin.router import router as admin_router
from src.calendar_events.router import router as calendar_events_router
from src.chat.router import router as chat_router
from src.documents.router import router as documents_router
from src.logs.router import router as logs_router
from src.professors.router import router as professors_router
from src.auth import get_api_key

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
# Autenticação própria (JWT de AdminUser), deliberadamente separada da x-api-key pública.
app.include_router(admin_router, prefix="/admin", tags=["Admin"])



@app.get("/health", tags=["Infra"])
async def health_check():
    """Verifica se a API está online."""
    return {"status": "ok"}
