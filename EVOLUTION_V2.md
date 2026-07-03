# EVOLUTION_V2.md — Registro de Evolução da Segunda Versão (TCC II)

> **Propósito:** Documento vivo, análogo ao [EVOLUTION.md](EVOLUTION.md) (que cobriu a v1 / TCC I), mas dedicado exclusivamente à segunda versão do backend. Serve como base para a escrita da monografia do TCC II. Atualize ao final de cada fase implementada, registrando o que foi feito, as decisões tomadas e o porquê.
>
> Documentos relacionados:
> - [PLANO_V2.md](PLANO_V2.md) — arquitetura e roadmap completo da v2 (10 fases), aprovado antes do início da implementação.
> - [EVOLUTION.md](EVOLUTION.md) — decisões D1-D9 da v1 (TCC I), ainda válidas e referenciadas aqui quando relevante.
>
> As decisões são numeradas em sequência com as de `EVOLUTION.md` (a última lá foi D9) — começam em **D10** — para manter um único histórico rastreável entre v1 e v2.

---

## Contexto Geral (herdado do TCC I, para não perder o fio)

A v1 entrega um pipeline Advanced RAG funcional sobre 48 documentos normativos institucionais da UNIVASF (Estatuto, Regimento Geral, Resoluções PROEN/PROEX/PRPPGI). O problema identificado ao final do TCC I: o agente consulta quase exclusivamente regulamentos institucionais, e faltam fontes de informação espalhadas em outros tipos de documento/dado (Manual do Aluno, Calendário Acadêmico, Corpo Docente), além de faltar infraestrutura administrativa para adicionar conteúdo sem depender de scripts CLI manuais.

A v2 ataca esse problema evoluindo a arquitetura de forma incremental — sem reescrever o pipeline de recuperação, que já funciona bem. O plano completo (10 fases) está em `PLANO_V2.md`; este documento registra a execução real, fase a fase.

---

## Fase 0 — Fundação de Dados (PostgreSQL + SQLAlchemy + Alembic)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** introduzir a única peça de infraestrutura genuinamente nova da v2 — um banco relacional — como pré-requisito para todas as fases seguintes (API administrativa, professores, escopo por curso). Sem tocar em nenhuma rota da API existente.

### O que foi feito

- Adicionadas as dependências `sqlalchemy`, `alembic`, `psycopg2-binary` ([pyproject.toml](pyproject.toml)).
- Modelagem de 9 entidades em [src/db/models.py](src/db/models.py) (SQLAlchemy 2.0, estilo `Mapped`/`mapped_column`, um único módulo): `Course`, `KnowledgeBase`, `Document`, `DocumentChunk`, `IngestionJob`, `Professor`, `Discipline`, `ProfessorDiscipline`, `AdminUser`.
- Camada de sessão [src/db/session.py](src/db/session.py) (`engine`, `SessionLocal`, dependência FastAPI `get_db()`).
- `DATABASE_URL` centralizado em [src/config.py](src/config.py), seguindo o mesmo padrão já usado para `CHROMA_HOST`/`CHROMA_PORT`.
- Alembic configurado ([alembic/env.py](alembic/env.py)) lendo `DATABASE_URL` e `Base.metadata` da aplicação em vez de duplicar configuração no `.ini`. Migration inicial gerada via `--autogenerate` e aplicada — 9 tabelas confirmadas no Postgres.
- Serviço `postgres` adicionado ao [docker-compose.yml](docker-compose.yml), replicando o padrão já usado para o `chromadb` (healthcheck, volume nomeado, rede `app-network`).
- `scripts/seed_db.py` — cria o curso `ENGCOMP` (Engenharia de Computação) e a knowledge base `regulamentos`. Idempotente.
- `scripts/backfill_documents.py` — registra no banco os 48 PDFs já processados pela v1 (varrendo `regimentos_estatutos_resolucoes/`, reaproveitando `classify_document`/`is_revoked_by_filename` de `src/etl/pdf_converter.py`). Não popula `DocumentChunk` (isso fica para a Fase 1).

### Decisões de Arquitetura

#### D10 — Um único módulo `src/db/models.py` (em vez de um pacote com um arquivo por entidade)

**Decisão:** todas as 9 entidades ficam em um único arquivo.

**Justificativa:** o projeto já segue o padrão de módulos enxutos (cada domínio em `src/` tem 1-2 arquivos: `router.py`, `service.py`, `schemas.py`). Nove entidades não justificam a sobrecarga de navegação de nove arquivos separados. Pode ser dividido depois se o schema crescer muito — decisão reversível e de baixo custo.

#### D11 — PostgreSQL via Docker Compose, com `create_engine` preguiçoso

**Decisão:** Postgres roda como serviço Docker (não SQLite embutido), mas a conexão só é aberta no primeiro uso — nenhum módulo que não importe `src.db` é afetado pela disponibilidade do banco.

**Justificativa:** já detalhada em `PLANO_V2.md` §7 (comparação Postgres vs. SQLite). O ponto novo, validado na implementação: `create_engine()` do SQLAlchemy não conecta no import, então a adição do banco é comprovadamente não-invasiva (ver seção de Verificação abaixo) — importante para o TCC justificar que a evolução foi *aditiva*, não uma reescrita.

#### D12 — Backfill cria `Document`, mas não `DocumentChunk`, nesta fase

**Decisão:** o script de backfill popula apenas o registro administrativo do documento (`Document`, status `indexed`), sem tentar reconstruir o mapeamento `chroma_id ↔ chunk` retroativamente.

**Justificativa:** mapear os `chroma_id`s exigiria reproduzir exatamente a lógica de geração de ID de `src/indexing/vector_store.py` (incluindo o sufixo de desambiguação `__{i}` em caso de colisão), o que só faz sentido implementar junto da lógica de reindexação/limpeza de órfãos (Fase 1 — `IngestionService`). Popular pela metade agora criaria uma falsa sensação de completude.

#### D13 — Deduplicação do backfill por `storage_path`, não por `filename`

**Decisão:** o critério de idempotência do backfill (evitar recriar o mesmo documento em execuções repetidas) usa o caminho relativo do arquivo, não o nome do arquivo isoladamente.

**Justificativa:** ver achado técnico abaixo — dois arquivos físicos distintos podem ter o mesmo nome em pastas diferentes. Usar `filename` como chave gerou `MultipleResultsFound` na segunda execução do script; `storage_path` é realmente único por construção do filesystem.

### Achado técnico (relevante para a seção de Limitações/Trabalhos Futuros do TCC)

Durante o backfill, dois PDFs com o **mesmo nome de arquivo** foram encontrados em pastas diferentes: `PROEN/resolucao-n-03-2022_curricularizao-da-extenso-na-univasf-pdf-nuvem-univasf.pdf` e `PROEX/resolucao-n-03-2022_curricularizao-da-extenso-na-univasf-pdf-nuvem-univasf.pdf`. Como `save_chunks()` em `src/chunking/legal_chunker.py` nomeia o JSONL de saída apenas pelo `filename` (sem incluir a categoria/pasta), a segunda execução do ETL da v1 **sobrescreveu silenciosamente** o arquivo de chunks da primeira. Resultado prático: hoje só uma das duas cópias tem vetores vivos no ChromaDB, embora ambas estejam fisicamente presentes no corpus.

Isso é um bug real e pré-existente da v1, não introduzido pela v2 — descoberto como efeito colateral de tornar os documentos rastreáveis em banco. Fica registrado aqui como **motivação concreta** (não apenas teórica) para a Fase 1, que deve nomear artefatos de ingestão por `document_id` em vez de por nome de arquivo.

### Verificação realizada

1. `docker compose up -d postgres` — subiu isolado, sem exigir API nem ChromaDB.
2. `alembic upgrade head` — 9 tabelas + `alembic_version` confirmadas via `psql \dt`.
3. `python scripts/seed_db.py` executado 2x — segunda execução não duplicou (idempotência confirmada).
4. `python scripts/backfill_documents.py` executado 2x — 48 documentos criados na primeira, 0 criados/48 já existentes na segunda.
5. **Prova de não-invasividade:** com o container do Postgres **parado**, a API subiu normalmente e uma chamada real a `POST /chat/` retornou 200 com resposta correta — evidência de que a fundação de dados foi uma adição pura, sem acoplar o caminho crítico do chat a uma dependência nova.
6. Após reiniciar o Postgres, os dados persistiram corretamente (48 documentos, 1 curso, 1 knowledge base).

### Estado atual da v2

| Fase | Descrição | Status |
|---|---|---|
| 0 | Fundação de dados (Postgres + SQLAlchemy + Alembic) | ✅ Concluída |
| 1 | Refatoração da ingestão (`IngestionService`, corrige staleness do BM25 e chunks órfãos) | 🔜 Próxima |
| 2 | API administrativa de documentos (upload, reindex, auth de admin) | ⏳ Pendente |
| 3 | Corpo docente (dados estruturados + Tool) | ⏳ Pendente |
| 4 | Orquestração multi-tool (function calling nativo) | ⏳ Pendente |
| 5 | Expansão de conteúdo (Manual do Aluno etc.) | ⏳ Pendente |
| 6 | Escopo por curso | ⏳ Pendente |
| 7 | Calendário acadêmico (condicional) | ⏳ Pendente |
| 8 | Observabilidade admin (opcional) | ⏳ Pendente |

### Nota para a escrita do TCC II

Esta fase é material natural para a seção de **Metodologia/Implementação** — em particular, o argumento de que a evolução foi *incremental e não-invasiva* (item 5 da verificação) é uma evidência concreta e testável para defender a decisão arquitetural de "evoluir, não reescrever" descrita em `PLANO_V2.md` §3. O achado do bug de colisão de nomes (acima) é um bom exemplo de descoberta empírica durante a implementação — vale registrar como validação de que a modelagem administrativa (banco relacional) tem valor mesmo antes da API admin existir, pois já expôs uma falha de dados que passaria despercebida em produção.

---

## Itens Pendentes / Próximos Passos

> Atualizar ao final de cada fase.

- [ ] Planejar e implementar Fase 1 (Ingestion Service): extrair lógica de `run_etl.py`/`run_indexing.py`, corrigir staleness do índice BM25 (D1 do diagnóstico em `PLANO_V2.md`), corrigir limpeza de chunks órfãos (D2), e corrigir a colisão de nomes descrita acima.
- [ ] Ativar `score_threshold` no reranker (D3 do diagnóstico) — correção pontual, pode ser feita a qualquer momento, independente das fases.
