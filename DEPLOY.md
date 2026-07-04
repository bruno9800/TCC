# Guia de Deploy

Desde a v2 (ver `PLANO_V2.md`/`EVOLUTION_V2.md`), o sistema não é mais só "ChromaDB + API" — tem também PostgreSQL (cursos, documentos, professores, disciplinas, calendário, admins), migrations via Alembic, e um schema de autenticação de admin (JWT) separado da `x-api-key` pública. Este guia cobre os dois cenários reais de deploy:

- **Opção A — Migrar de uma instância existente** (recomendado quando você já tem o sistema rodando em algum lugar — ex: sua máquina de desenvolvimento — e quer replicá-lo em outra, preservando todo o corpus já indexado, professores, disciplinas e calendário sem reprocessar nada nem gastar API de embeddings de novo).
- **Opção B — Nova instalação do zero** (sem dados prévios).

Se você está saindo de uma máquina que já tem o corpus, o corpo docente e o calendário funcionando (o caso mais comum — ex: subir o mesmo projeto num desktop novo), **use a Opção A**. Ela é mais rápida, mais barata (não re-chama a API de embeddings da OpenAI) e evita o gap descrito na seção 2.3 abaixo.

> Todos os comandos deste guia têm um alvo equivalente no `Makefile` (`make help` lista todos). Ex.: a Opção A vira `make backup` na origem e `make restore` no destino; a Opção B vira `make bootstrap`. Os blocos de comando abaixo continuam documentados passo a passo para quem quiser entender ou adaptar o que cada `make` faz por baixo dos panos.

---

## 1. Pré-requisitos (ambas as opções)

```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER && newgrp docker
```

Confirme que `docker compose version` funciona (Compose v2, integrado ao Docker — não precisa instalar separado nas versões recentes).

---

## Opção A — Migrar de uma Instância Existente

Move os dados de duas origens diferentes — **não é tudo volume Docker**:

- `postgres_data` e `chroma_data` (cursos/documentos/professores/disciplinas/calendário/admins e os vetores) já são volumes Docker de verdade, populados pelos containers `postgres`/`chromadb` usados durante o desenvolvimento.
- `data/raw/` (PDFs enviados), `data/chunks/` (JSONLs usados pelo BM25) e `data/logs/` (histórico de queries) — se você rodou o projeto localmente até agora (`uvicorn` direto, fora do container `api`, que é o padrão usado durante o desenvolvimento deste projeto), **esses três vivem no disco do host**, na própria pasta do projeto, não em volume Docker. Isso importa em especial para `data/chunks/`: está tanto no `.gitignore` quanto no `.dockerignore` (gerado, não versionado — mas legitimamente necessário em runtime para o BM25), então **nem `git clone` nem `docker build` o transferem** — sem copiar esse diretório manualmente, o container novo sobe com o BM25 vazio (`HybridSearchEngine` degrada para busca só-densa, sem erro nenhum — silencioso). O `docker-compose.yml` já declara `data_raw`/`data_chunks`/`data_logs` como volumes persistentes para o serviço `api`, para que isso não se repita a cada rebuild *depois* da primeira migração.

### 1. Na máquina de origem — empacotar tudo

```bash
cd /caminho/do/projeto
docker compose stop   # garante consistência do postgres/chroma antes de copiar

mkdir -p ~/tcc-migration

# postgres_data e chroma_data: volumes Docker de verdade
for VOL in postgres_data chroma_data; do
  FULL_NAME=$(docker volume ls -q | grep "_${VOL}$")
  echo "Empacotando ${FULL_NAME}..."
  docker run --rm -v "${FULL_NAME}:/data" -v ~/tcc-migration:/backup alpine \
    tar czf "/backup/${VOL}.tar.gz" -C /data .
done

# data/raw, data/chunks, data/logs: no disco do host, não em volume Docker
tar czf ~/tcc-migration/data_raw.tar.gz    -C data/raw    .
tar czf ~/tcc-migration/data_chunks.tar.gz -C data/chunks .
tar czf ~/tcc-migration/data_logs.tar.gz   -C data/logs   .

docker compose start   # ou deixe parado, se for desligar a máquina
```

Transfira `~/tcc-migration/*.tar.gz` (5 arquivos) para a máquina destino (scp, pendrive, o que for prático).

### 2. Na máquina destino — clonar o código e restaurar os volumes

```bash
git clone https://github.com/bruno9800/TCC.git
cd TCC
cp .env.example .env
nano .env   # preencher OPENAI_API_KEY, TCC_API_KEY, JWT_SECRET (ver seção 2.4), TUNNEL_TOKEN

docker compose up -d --build   # cria os volumes vazios e a rede
docker compose stop            # para poder escrever nos volumes sem conflito de processo

for VOL in postgres_data chroma_data data_raw data_chunks data_logs; do
  FULL_NAME=$(docker volume ls -q | grep "_${VOL}$")
  docker run --rm -v "${FULL_NAME}:/data" -v ~/tcc-migration:/backup alpine \
    sh -c "rm -rf /data/* && tar xzf /backup/${VOL}.tar.gz -C /data"
done

docker compose up -d
docker compose exec api alembic upgrade head   # no-op se o schema já estiver atualizado; garante consistência se o código clonado for mais novo que o volume
```

### 3. Verificar

```bash
curl http://localhost:8000/health
curl -H "x-api-key: <sua-chave>" http://localhost:8000/courses
curl -H "x-api-key: <sua-chave>" -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Quando é o período de trancamento de matrícula em 2026.1?"}'
```

Se a última chamada responder com a data exata (via `search_academic_calendar`), o calendário migrou corretamente junto com o Postgres. Confira também que `GET /professors` e `GET /courses` retornam o corpo docente e o(s) curso(s) esperados.

---

## Opção B — Nova Instalação (sem dados prévios)

### 1. Clonar e configurar

```bash
git clone https://github.com/bruno9800/TCC.git
cd TCC
cp .env.example .env
nano .env   # preencher OPENAI_API_KEY, TCC_API_KEY, JWT_SECRET, TUNNEL_TOKEN (ver seção 2.4)
```

> Manter `CHROMA_HOST=chromadb` e `CHROMA_PORT=8000` no `.env` — o serviço `api` do `docker-compose.yml` já sobrescreve isso automaticamente para o ambiente Docker; esses valores no `.env` só importam para rodar fora do Docker (seção 5).

### 2. Subir os serviços e aplicar o schema

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
```

### 3. Popular os dados base

```bash
# Curso (ENGCOMP) + Knowledge Base (regulamentos) — obrigatório, tudo mais depende disso
docker compose exec api python scripts/seed_db.py

# Primeiro admin, para acessar /admin/*
docker compose exec api python scripts/create_admin.py admin@univasf.edu.br "sua-senha-forte"

# Corpo docente, matriz curricular e calendário real (idempotentes — seguro rodar mais de uma vez)
docker compose exec api python scripts/seed_professors_engcomp.py
docker compose exec api python scripts/seed_disciplines_engcomp.py
docker compose exec api python scripts/seed_calendar_2026.py
```

### 3.1 Indexar o corpus normativo (Estatuto, Regimento Geral, resoluções, PPC)

**Gap conhecido:** não existe hoje um script committed que pegue os PDFs de `regimentos_estatutos_resolucoes/` (50 arquivos, já versionados no git) e os processe do zero através do pipeline novo (ETL → chunk → embed → index). O script `scripts/backfill_documents.py` existente **assume que os vetores já estão no ChromaDB** (foi escrito para migrar metadados da v1, não para popular um ChromaDB vazio) — rodá-lo aqui registraria os documentos como `status="indexed"` sem, de fato, terem sido embedados, e o chat não encontraria nada.

Caminhos possíveis, até esse gap ser fechado com um script dedicado:

- **Preferível:** use a Opção A (migrar volumes de uma instância que já tem o corpus indexado) em vez desta seção.
- **Alternativa manual:** suba cada PDF via `POST /admin/documents` (ver `INTEGRACAO_FRONTEND.md`, seção 3.1) — o endpoint já faz ETL → chunk → embed → index por trás, um documento por vez. Viável para o PPC (1 arquivo), trabalhoso para os ~50 documentos institucionais.

### 4. Verificar

Mesmos comandos de verificação da Opção A (seção 1.3) — a única diferença é que, sem indexar o corpus (passo 3.1), perguntas normativas não vão encontrar nada; perguntas sobre professores/disciplinas/calendário já funcionam normalmente após o passo 3.

---

## 2. Configuração de Ambiente (`.env`)

### 2.1 Variáveis obrigatórias

| Variável | Descrição |
|---|---|
| `OPENAI_API_KEY` | Embeddings + geração (GPT-4o) |
| `TCC_API_KEY` | Protege `/chat`, `/documents`, `/logs`, `/professors`, `/academic-events`, `/courses`. Se vazia, essas rotas ficam **sem autenticação** — nunca em produção |

### 2.2 PostgreSQL

`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` (usadas pelo container `postgres`) e `DATABASE_URL` (usada pela app — em Docker, o serviço `api` já sobrescreve isso para apontar pro host `postgres`; o valor do `.env` só é usado rodando fora do Docker).

### 2.3 Admin (JWT)

`JWT_SECRET` — **defina um valor fixo em produção** (`openssl rand -hex 32`). Se deixado vazio, a aplicação gera um segredo aleatório em memória a cada start do processo — funciona, mas **invalida todos os tokens de admin emitidos a cada restart do container**, o que inclui todo `docker compose up -d --build` de um deploy. `JWT_EXPIRE_MINUTES` (padrão `480` = 8h).

### 2.4 Cloudflare Tunnel

`TUNNEL_TOKEN` — ver seção 3 abaixo. Se vazio, o container `tunnel` simplesmente não inicia (sem erro fatal para o resto do stack).

---

## 3. Cloudflare Tunnel (acesso público sem abrir portas)

Usar quando precisar expor a API com URL pública fixa (ex: para a banca ou testes remotos) sem mexer em firewall/roteador.

### 3.1 Obter o Token

1. Acesse o [Zero Trust Dashboard](https://one.dash.cloudflare.com/).
2. Vá em **Networks > Tunnels > Create a Tunnel**.
3. Escolha **Cloudflared**, dê um nome (ex: `tcc-api`).
4. Copie o **Token** exibido (string longa começando com `ey...`).

### 3.2 Configurar

No `.env`:

```bash
TUNNEL_TOKEN=eyJhIjoi...
```

### 3.3 Configurar Rota Pública no Cloudflare

Na aba **Public Hostname** do tunnel:
- **Subdomain**: `api` (ex: `api.seu-dominio.com`)
- **Domain**: seu domínio
- **Service**: `HTTP` → `univasf-api:8000` (nome do container Docker — resolve via a rede interna `app-network` do compose, não precisa ser IP)

### 3.4 Rodar

```bash
docker compose up -d
```

O container `univasf-tunnel` conecta automaticamente e a API fica acessível no domínio configurado — sem nenhuma porta aberta no roteador/firewall da máquina.

> Se o desktop estiver atrás de CGNAT ou rede doméstica sem IP público (comum em conexões residenciais), o Cloudflare Tunnel é a única forma prática de expor a API sem VPN — ele faz uma conexão de saída da máquina para o Cloudflare, não depende de porta de entrada.

---

## 4. Deploy Contínuo (Atualizar)

```bash
./scripts/deploy.sh
```

Faz `git pull` + rebuild dos containers + **aplica migrations pendentes do Alembic** (`alembic upgrade head`, adicionado nesta atualização do guia — antes disso, um `git pull` que trouxesse uma migration nova exigiria rodá-la manualmente, e um deploy "esquecido" deixaria o schema do Postgres defasado em relação ao código). Se o pull trouxe novo conteúdo para indexar, rode manualmente:

```bash
docker compose exec api python scripts/run_etl.py       # no-op se não houver documentos pendentes
docker compose exec api python scripts/run_indexing.py  # idem
```

(Ambos são seguros por padrão — só tocam documentos ainda não processados. Para forçar reprocessar um documento específico: `--reindex <document_id>`.)

---

## 5. Desenvolvimento Local (sem Docker)

```bash
# Instalar dependências
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Subir só o Postgres e o ChromaDB via Docker (mais simples que instalar localmente)
docker compose up -d postgres chromadb

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env: OPENAI_API_KEY, TCC_API_KEY, JWT_SECRET
# CHROMA_HOST=localhost, CHROMA_PORT=8001 (porta mapeada do container, ver docker-compose.yml)
# DATABASE_URL=postgresql+psycopg2://univasf:univasf@localhost:5432/univasf (porta mapeada 5432)

# Aplicar migrations
alembic upgrade head

# Popular dados base (mesma sequência da Opção B, seção 1.3, sem o prefixo `docker compose exec api`)
python scripts/seed_db.py
python scripts/create_admin.py admin@univasf.edu.br "sua-senha-forte"
python scripts/seed_professors_engcomp.py
python scripts/seed_disciplines_engcomp.py
python scripts/seed_calendar_2026.py

# Subir a API
uvicorn src.main:app --port 8000 --reload

# (Opcional) Avaliar com RAGAS
python scripts/run_eval.py
```
