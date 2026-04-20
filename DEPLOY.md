# Guia de Deploy

## Opção A — VPS (Ubuntu + Docker)

### 1. Pré-requisitos na VPS

```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER && newgrp docker
```

### 2. Configuração Inicial

```bash
git clone https://github.com/bruno9800/TCC.git
cd TCC
cp .env.example .env
nano .env   # Preencher OPENAI_API_KEY e TCC_API_KEY
```

> Manter `CHROMA_HOST=chromadb` e `CHROMA_PORT=8000` para o ambiente Docker.

### 3. Subir os Serviços

```bash
docker compose up -d --build
```

### 4. Indexar Documentos (primeira vez)

Os PDFs devem estar em `regimentos_estatutos_resolucoes/` antes deste passo.
O índice persiste no volume `chroma_data` — rodar apenas uma vez ou ao adicionar novos arquivos.

```bash
docker compose run --rm api python scripts/run_indexing.py
```

### 5. Atualizar (Deploy Contínuo)

```bash
./deploy.sh
```

O script `deploy.sh` faz `git pull` + rebuild dos containers + limpeza de imagens antigas.

---

## Opção B — Cloudflare Tunnel (acesso público sem abrir portas)

Usar quando precisar expor a API com URL pública fixa (ex: para a banca ou testes remotos).

### 1. Obter o Token

1. Acesse o [Zero Trust Dashboard](https://one.dash.cloudflare.com/).
2. Vá em **Networks > Tunnels > Create a Tunnel**.
3. Escolha **Cloudflared**, dê um nome (ex: `tcc-api`).
4. Copie o **Token** exibido (string longa começando com `ey...`).

### 2. Configurar

No `.env`, adicione:

```bash
TUNNEL_TOKEN=eyJhIjoi...
```

### 3. Configurar Rota Pública no Cloudflare

Na aba **Public Hostname** do tunnel:
- **Subdomain**: `api` (ex: `api.seu-dominio.com`)
- **Domain**: seu domínio
- **Service**: `HTTP` → `univasf-api:8000` (nome do container Docker)

### 4. Rodar

```bash
docker compose up -d
```

O container `univasf-tunnel` conecta automaticamente e a API fica acessível no domínio configurado.

---

## Desenvolvimento Local (sem Docker)

```bash
# Instalar dependências
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env: OPENAI_API_KEY, TCC_API_KEY, CHROMA_HOST=localhost

# Rodar ETL (PDF → chunks)
python scripts/run_etl.py

# Indexar no ChromaDB
python scripts/run_indexing.py

# Subir API
uvicorn src.main:app --port 8000 --reload

# (Opcional) Avaliar com RAGAS
python scripts/run_eval.py
```
