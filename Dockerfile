# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# Evita prompts interativos e buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dependências do sistema ────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Dependências Python ───────────────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Código da aplicação ───────────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY regimentos_estatutos_resolucoes/ ./regimentos_estatutos_resolucoes/

# ── Dados pré-processados (chunks) ────────────────────────────────────────────
# Copia dados de chunks se existirem (necessários para o BM25 do HybridSearchEngine)
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
