# Automação do deploy descrito em DEPLOY.md.
# Rode `make help` (ou apenas `make`) para a lista de comandos.

COMPOSE    ?= docker compose
BACKUP_DIR ?= $(HOME)/tcc-migration
VOLUMES    := postgres_data chroma_data data_raw data_chunks data_logs

.DEFAULT_GOAL := help
.PHONY: help check-env build up down restart ps logs migrate seed create-admin \
        reindex bootstrap deploy backup restore health sync-chunks

help:
	@echo "UNIVASF RAG API — alvos de deploy (ver DEPLOY.md para o passo a passo detalhado)"
	@echo ""
	@echo "  Instalação nova (Opção B):"
	@echo "    make bootstrap            up + migrate + seed (curso/KB, corpo docente, disciplinas, calendário)"
	@echo "    make create-admin EMAIL=admin@univasf.edu.br PASSWORD=senha-forte"
	@echo ""
	@echo "  Migrar de uma instância existente (Opção A, recomendado):"
	@echo "    make backup               empacota volumes + data/raw,chunks,logs em \$$BACKUP_DIR (máquina de origem)"
	@echo "    make restore              restaura o backup nos volumes (máquina destino — destrutivo, pede confirmação)"
	@echo "    BACKUP_DIR=/caminho make backup|restore   para usar outro diretório (padrão: ~/tcc-migration)"
	@echo ""
	@echo "  Operação do dia a dia:"
	@echo "    make up                   docker compose up -d --build"
	@echo "    make down                 docker compose down"
	@echo "    make deploy               git pull + rebuild + migrations (== ./scripts/deploy.sh)"
	@echo "    make migrate              alembic upgrade head"
	@echo "    make reindex              processa documentos pendentes (run_etl.py + run_indexing.py; no-op se não houver)"
	@echo "    make logs                 segue os logs do container api"
	@echo "    make ps                   status dos containers"
	@echo "    make health               checa GET /health"

check-env:
	@test -f .env || (echo "❌ .env não encontrado. Rode: cp .env.example .env && nano .env" && exit 1)

build: check-env
	$(COMPOSE) build

up: check-env
	$(COMPOSE) up -d --build
	@$(MAKE) sync-chunks

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart api

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f api

migrate:
	$(COMPOSE) exec api alembic upgrade head

# Dado real, idempotente — seguro rodar mais de uma vez (ver scripts/seed_*.py).
seed: migrate
	$(COMPOSE) exec api python scripts/seed_db.py
	$(COMPOSE) exec api python scripts/seed_professors_engcomp.py
	$(COMPOSE) exec api python scripts/seed_disciplines_engcomp.py
	$(COMPOSE) exec api python scripts/seed_calendar_2026.py

create-admin: check-env
	@test -n "$(EMAIL)" && test -n "$(PASSWORD)" || \
		(echo "uso: make create-admin EMAIL=admin@univasf.edu.br PASSWORD=senha-forte" && exit 1)
	$(COMPOSE) exec api python scripts/create_admin.py "$(EMAIL)" "$(PASSWORD)"

# Documentos pendentes de ETL/indexação (status processing/failed/chunked) — no-op
# por padrão, seguro rodar a qualquer momento (ver scripts/run_etl.py, run_indexing.py).
reindex:
	$(COMPOSE) exec api python scripts/run_etl.py
	$(COMPOSE) exec api python scripts/run_indexing.py

bootstrap: up migrate seed
	@echo ""
	@echo "✅ Instalação do zero completa."
	@echo "   Próximo passo: make create-admin EMAIL=... PASSWORD=..."
	@echo "   Gap conhecido: o corpus normativo (Estatuto/Regimento/PPC) ainda precisa"
	@echo "   ser indexado manualmente — ver DEPLOY.md, Opção B, seção 3.1."

deploy:
	./scripts/deploy.sh

health:
	@curl -sf http://localhost:8000/health && echo " OK" || (echo " FALHOU" && exit 1)

# Copia data/chunks/ do disco do host (populado quando o projeto roda localmente
# via `uvicorn`, fora do Docker) para o volume `data_chunks` do serviço `api`.
# Necessário na PRIMEIRA vez que se passa a rodar via container nesta mesma
# máquina: esse diretório está no .dockerignore (nunca entra na imagem), então
# o volume novo sobe vazio e o BM25 degrada pra busca só-densa, sem erro nenhum.
# Chamado automaticamente pelo `up` — no-op silencioso se não houver nada a
# copiar (ex: instalação nova, sem histórico local).
sync-chunks:
	@if [ -d data/chunks ] && [ -n "$$(ls -A data/chunks 2>/dev/null)" ]; then \
		FULL_NAME=$$(docker volume ls -q | grep "_data_chunks$$"); \
		echo "Copiando data/chunks/ (host) → volume $$FULL_NAME..."; \
		docker run --rm -v "$$FULL_NAME:/dest" -v "$(CURDIR)/data/chunks:/src:ro" alpine \
			sh -c "cp -a /src/. /dest/"; \
		$(COMPOSE) restart api; \
		echo "✅ Chunks sincronizados, BM25 recarregado."; \
	fi

# ── Migração entre máquinas (Opção A do DEPLOY.md) ──────────────────────────
#
# postgres_data/chroma_data são volumes Docker de verdade. data/raw, data/chunks
# e data/logs vivem no disco do host quando o projeto é rodado via `uvicorn`
# local (padrão de desenvolvimento) — data/chunks em particular está tanto no
# .gitignore quanto no .dockerignore, então nem `git clone` nem `docker build`
# o levam para a máquina destino; sem copiá-lo manualmente, o BM25 sobe vazio
# e a busca degrada para só-densa, silenciosamente.

backup: check-env
	@mkdir -p $(BACKUP_DIR)
	$(COMPOSE) stop
	@for VOL in postgres_data chroma_data; do \
		FULL_NAME=$$(docker volume ls -q | grep "_$${VOL}$$"); \
		echo "Empacotando $$FULL_NAME..."; \
		docker run --rm -v "$$FULL_NAME:/data" -v $(BACKUP_DIR):/backup alpine \
			tar czf "/backup/$${VOL}.tar.gz" -C /data . ; \
	done
	tar czf $(BACKUP_DIR)/data_raw.tar.gz    -C data/raw    .
	tar czf $(BACKUP_DIR)/data_chunks.tar.gz -C data/chunks .
	tar czf $(BACKUP_DIR)/data_logs.tar.gz   -C data/logs   .
	$(COMPOSE) start
	@echo ""
	@echo "✅ Backup em $(BACKUP_DIR) (5 arquivos). Transfira para a máquina destino"
	@echo "   (scp/pendrive) e rode 'make restore BACKUP_DIR=<mesmo-caminho-la>'."

restore: check-env
	@test -d "$(BACKUP_DIR)" || (echo "❌ BACKUP_DIR '$(BACKUP_DIR)' não encontrado" && exit 1)
	@echo "⚠️  Isso APAGA os dados atuais dos volumes antes de restaurar o backup."
	@read -p "Continuar? [y/N] " ans; [ "$$ans" = "y" ] || (echo "Cancelado." && exit 1)
	$(COMPOSE) up -d --build
	$(COMPOSE) stop
	@for VOL in $(VOLUMES); do \
		FULL_NAME=$$(docker volume ls -q | grep "_$${VOL}$$"); \
		echo "Restaurando $$FULL_NAME..."; \
		docker run --rm -v "$$FULL_NAME:/data" -v $(BACKUP_DIR):/backup alpine \
			sh -c "rm -rf /data/* && tar xzf /backup/$${VOL}.tar.gz -C /data"; \
	done
	$(COMPOSE) up -d
	$(MAKE) migrate
	@echo "✅ Restauração concluída — rode 'make health' para confirmar."
