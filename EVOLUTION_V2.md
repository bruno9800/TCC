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
| 1 | Refatoração da ingestão (`IngestionService`, corrige staleness do BM25 e chunks órfãos) | ✅ Concluída |
| 2 | API administrativa de documentos (upload, reindex, auth de admin) | 🔜 Próxima |
| 3 | Corpo docente (dados estruturados + Tool) | ⏳ Pendente |
| 4 | Orquestração multi-tool (function calling nativo) | ⏳ Pendente |
| 5 | Expansão de conteúdo (Manual do Aluno etc.) | ⏳ Pendente |
| 6 | Escopo por curso | ⏳ Pendente |
| 7 | Calendário acadêmico (condicional) | ⏳ Pendente |
| 8 | Observabilidade admin (opcional) | ⏳ Pendente |

### Nota para a escrita do TCC II

Esta fase é material natural para a seção de **Metodologia/Implementação** — em particular, o argumento de que a evolução foi *incremental e não-invasiva* (item 5 da verificação) é uma evidência concreta e testável para defender a decisão arquitetural de "evoluir, não reescrever" descrita em `PLANO_V2.md` §3. O achado do bug de colisão de nomes (acima) é um bom exemplo de descoberta empírica durante a implementação — vale registrar como validação de que a modelagem administrativa (banco relacional) tem valor mesmo antes da API admin existir, pois já expôs uma falha de dados que passaria despercebida em produção.

---

## Fase 1 — Ingestion Service (extração + correção de D1/D2)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** extrair a lógica de ingestão hoje espalhada em `run_etl.py`/`run_indexing.py` para um serviço reaproveitável (`src/ingestion/service.py`), usado tanto pelos scripts CLI quanto, na Fase 2, pela futura API de upload. Corrigir dois bugs diagnosticados em `PLANO_V2.md` (D1: staleness do índice BM25; D2: chunks órfãos em reindex) e a colisão de nomes de arquivo encontrada durante o backfill da Fase 0.

### O que foi feito

- `ChunkMetadata`/`chunk_document()` ([src/chunking/legal_chunker.py](src/chunking/legal_chunker.py)) ganharam `kb_slug`/`course_id`, propagados também nas duas construções manuais dentro de `split_long_chunk()`.
- `index_chunks()` ([src/indexing/vector_store.py](src/indexing/vector_store.py)) passou a aceitar `id_prefix` (namespace por `document_id`, ex. `doc42__...`) e a **retornar a lista de IDs gerados** (antes retornava a `Collection`) — necessário para popular `DocumentChunk`. `flat_meta` ganhou `kb_slug`/`course_id` (sentinela `0` para institucional, já que ChromaDB não aceita `None`).
- `HybridSearchEngine.reload()` ([src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py)) — recarrega `self.chunks` e reconstrói o BM25, fechando D1. O singleton `get_search_engine()` foi **relocado** de `src/chat/service.py` para cá (é sobre o motor de recuperação, não sobre chat; evita que `src/ingestion/` dependesse de `src/chat/` só para invalidar um cache).
- Novo módulo `src/ingestion/service.py`: `etl_and_chunk()`, `embed_and_index()` (limpa vetores antigos rastreados via `DocumentChunk` antes de reindexar — fecha D2 — e chama `get_search_engine().reload()` ao final), e `process_document()` (composição das duas, para a futura API de upload). Bookkeeping em `IngestionJob` (tabela criada na Fase 0, sem uso até agora).
- `scripts/run_etl.py`/`run_indexing.py` reescritos como wrappers finos, DB-driven: sem argumentos, só processam documentos pendentes (`status` `processing`/`failed` ou `chunked`, respectivamente) — **no-op seguro por padrão**. Ganharam `--reindex <document_id>` para forçar o reprocessamento de um documento específico.

### Decisões de Arquitetura

#### D14 — Manter `etl_and_chunk`/`embed_and_index` separados (não uma função única obrigatória)

**Decisão:** o serviço expõe duas funções componíveis, além de `process_document()` que as encadeia.

**Justificativa:** os dois scripts da v1 já eram separados por um motivo real — permitir inspecionar os chunks gerados antes de pagar pelos embeddings. Colapsar tudo em uma função única removeria esse workflow sem necessidade. `process_document()` existe para quem só quer o pipeline completo (caso da futura API de upload).

#### D15 — IDs de chunk namespaced por `document_id`, não por `filename`

**Decisão:** `index_chunks(..., id_prefix=f"doc{document.id}")` — o ID do chunk no ChromaDB e o nome do JSONL (`doc{id}.jsonl`) passam a derivar da chave primária do documento no banco, não do nome do arquivo.

**Justificativa:** correção estrutural do bug encontrado na Fase 0 (dois PDFs com o mesmo `filename`, em pastas diferentes, sobrescrevendo o JSONL um do outro). Como `document_id` é único por construção (chave primária), essa classe inteira de bug deixa de ser possível para qualquer documento processado pelo novo serviço.

#### D16 — Limpeza de vetores órfãos só para documentos já rastreados via `DocumentChunk`; sem "limpeza automática por metadado" no caminho padrão

**Decisão:** `embed_and_index()` só apaga vetores antigos quando existem linhas de `DocumentChunk` para aquele `document_id` (deleção precisa, por `chroma_id`). Documentos sem `DocumentChunk` (ainda não migrados para o novo esquema) **não** disparam uma limpeza automática por filtro de metadado `source`.

**Justificativa:** o campo `source` não é garantidamente único — é justamente a causa da colisão da Fase 0. Uma limpeza automática por `source` no caminho padrão arriscaria apagar vetores de **outro** documento já migrado que compartilhe o mesmo título (exatamente o caso real de `doc34`/`doc42`, encontrado ao planejar a verificação desta fase, antes de rodar qualquer coisa em produção). A migração inicial de artefatos legados é feita uma única vez, manualmente e sob supervisão — documentada abaixo — e depois disso todo documento tem `DocumentChunk` e cai no caminho seguro.

#### D17 — Scripts CLI seguros por padrão, com `--reindex` como escape hatch explícito

**Decisão:** `run_etl.py`/`run_indexing.py`, sem argumentos, só tocam documentos ainda não processados. Reprocessar um documento já indexado exige `--reindex <id>` explícito.

**Justificativa:** evita que rodar os scripts "sem pensar" reprocesse (e gaste embeddings de) todo o corpus já indexado. É também o mesmo mecanismo que a Fase 2 (`POST /admin/documents/{id}/reindex`) vai acionar por baixo dos panos — validado aqui antes de existir API.

### Achados técnicos e decisão de escopo ampliada durante a verificação

Ao planejar a verificação (reindexar apenas os documentos 34/42 para corrigir o achado da Fase 0), esbarrei em um problema pré-existente e independente desta fase: **o vectorstore local (`data/vectorstore/`) estava ilegível** pelo cliente ChromaDB pinado no projeto (`0.6.3`, mesma versão da imagem Docker). O erro (`KeyError: '_type'` ao listar collections) indica que os dados em disco foram gravados por uma versão mais nova do cliente (`1.5.0`, que estava instalada no `.venv` antes da Fase 0 reinstalar as dependências conforme o pin do `pyproject.toml`) — as duas versões usam formatos de configuração de collection incompatíveis. Além disso, o volume Docker do `chromadb` nunca tinha sido populado (volume novo, vazio).

Como os chunks fonte (`data/chunks/*.jsonl`) continuavam íntegros, a saída não destrutiva era regenerar o índice vetorial a partir deles. Apresentei a situação ao usuário com três opções (reindexar tudo, reindexar só 34/42, ou pausar) — **decisão do usuário: reindexar o corpus inteiro agora**, usando o novo `IngestionService` contra o ChromaDB do Docker (recém-criado e vazio). Resultado:

- 48/48 documentos migrados com sucesso (0 falhas), ~1217 chunks/vetores no total.
- Os 47 JSONLs antigos (nomeados por filename) foram removidos após a migração — sem isso, `load_all_chunks()` carregava tanto os arquivos antigos quanto os novos `doc{id}.jsonl`, duplicando o corpus do BM25 (2433 chunks carregados vs. 1217 no Chroma, antes da limpeza).
- `documents 34` e `42` (o achado da Fase 0) confirmadamente passaram a ter vetores distintos e corretos: `doc34__preamble__0` e `doc42__preamble__0` — o bug está fechado de fato, não só em tese.
- `.env` local tinha `CHROMA_HOST=chromadb` (hostname que só resolve dentro da rede Docker) — quebra qualquer script rodado direto no host fora de containers. Corrigido para `localhost` (o `docker-compose.yml` sobrescreve essa variável para `chromadb` especificamente dentro do container `api`, então o deploy via Docker não é afetado).

Este achado (drift de versão de uma dependência de terceiros silenciosamente corrompendo dados locais) é um bom exemplo para a seção de **Limitações/Discussão** do TCC II — reforça o valor de pinar versões explicitamente (`pyproject.toml` já fazia isso) e de ter a estratégia de regeneração a partir de uma fonte de verdade (`data/chunks/`) como rede de segurança.

### Verificação realizada

1. Sanity import de todos os módulos tocados — sem erro.
2. `run_etl.py`/`run_indexing.py` sem argumentos — no-op quando não há documentos pendentes (confirmado antes e depois da migração completa).
3. Migração completa dos 48 documentos via `IngestionService` (script pontual, não versionado) — 0 falhas.
4. Contagens cruzadas batendo: 1217 `document_chunks` no Postgres == 1217 vetores no ChromaDB == 1217 chunks carregados do disco para o BM25 (após remover os JSONLs legados).
5. Confirmação direta no ChromaDB de que `doc34`/`doc42` têm vetores próprios e distintos.
6. `POST /chat/` real, pergunta sobre trancamento de matrícula → 200, busca acionada, resposta corretamente fundamentada no Art. 72 da Resolução 08/2015 com citação de §1º/§2º — pipeline de chat funcionando ponta a ponta com o `get_search_engine()` relocado e o corpus migrado.

### Nota para a escrita do TCC II

O achado do vectorstore corrompido e a decisão de migrar o corpus inteiro (em vez de só os dois documentos originalmente escopados) são um bom exemplo de como o escopo de uma fase pode mudar legitimamente diante de uma descoberta em produção — vale contrastar com a Fase 0, que foi executada exatamente como planejado. Documentar essa diferença fortalece a seção de metodologia (DSR previsivelmente itera).

---

## Itens Pendentes / Próximos Passos

> Atualizar ao final de cada fase.

- [ ] Planejar e implementar Fase 2 (API administrativa de documentos): upload, reindex via HTTP, auth de `AdminUser`.
- [ ] Ativar `score_threshold` no reranker (D3 do diagnóstico) — correção pontual, pode ser feita a qualquer momento, independente das fases.
