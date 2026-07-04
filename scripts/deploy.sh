#!/bin/bash

# Script de Automação de Deploy
# Uso: ./deploy.sh
# Requer: git, docker, docker compose

set -e # Para o script se houver erro

echo "========================================"
echo "🚀 Iniciando Deploy do UNIVASF RAG"
echo "========================================"

# 1. Verifica se está na pasta correta
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Erro: docker-compose.yml não encontrado."
    echo "Execute este script na raiz do projeto."
    exit 1
fi

# 2. Atualiza Código
echo "📥 Baixando atualizações..."
git pull origin main

# 3. Reconstrói e Reinicia Containers
echo "🐳 Atualizando containers Docker..."
# --build: Força reconstrução da imagem se houver mudanças no Dockerfile/requirements
# --remove-orphans: Remove containers que não estão mais no docker-compose
docker compose up -d --build --remove-orphans

# 4. Aplica migrations pendentes
# Necessário sempre que o pull trouxer uma nova migration (alembic/versions/) —
# sem isso, o schema do Postgres fica defasado em relação ao código e a API
# quebra em runtime na primeira query que tocar uma coluna/tabela nova.
echo "🗄️  Aplicando migrations do banco..."
docker compose exec -T api alembic upgrade head

# 5. Verifica status
echo "🔍 Verificando serviços..."
sleep 5
docker compose ps

echo "========================================"
echo "✅ Deploy finalizado!"
echo "========================================"
echo "Se você adicionou novos documentos, lembre-se de rodar:"
echo "docker compose run --rm api python scripts/run_indexing.py"
