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
| 2 | API administrativa de documentos (upload, reindex, auth de admin) | ✅ Concluída |
| 3 | Corpo docente (dados estruturados + Tool) | ✅ Concluída (dados/CRUD — Tool é Fase 4) |
| 4 | Orquestração multi-tool (function calling nativo) | ✅ Concluída |
| 5a | PPC real (RAG + matriz curricular estruturada) | ✅ Concluída |
| 5b | Calendário Acadêmico (dado estruturado + Tool) | ✅ Concluída |
| 6 | Escopo por curso | ⏳ Pendente |
| 7 | Calendário acadêmico (condicional) | ✅ Absorvida pela Fase 5b (dado real ficou disponível antes da condição de demanda validada) |
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

## Fase 2 — API Administrativa de Documentos

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** expor a infraestrutura das Fases 0/1 via HTTP — upload, listagem, atualização, remoção e reindexação de documentos — protegida por autenticação de administrador (JWT), deliberadamente separada da `x-api-key` pública que já protege `/chat`, `/documents`, `/logs`.

### O que foi feito

- `src/admin/` (novo módulo): `auth.py` (hash/verificação de senha via `bcrypt`, `create_access_token`, `get_current_admin` — dependência FastAPI via `HTTPBearer`, espelhando o padrão `Security`/`Depends` já usado em `src/auth.py`), `schemas.py`, `router.py` (`POST /admin/auth/login` + CRUD de documentos).
- `src/documents/service.py` (novo — `DocumentService`): `create_document` (salva o arquivo em `data/raw/{id}/`, dispara `IngestionService.process_document` da Fase 1 — se a ingestão falhar, o documento fica com `status="failed"` mas o upload em si não retorna erro), `list_documents`, `get_document`, `update_document` (semântica PATCH via `exclude_unset`), `reindex_document`, `delete_document`.
- `src/documents/router.py`: `GET /documents/list` e `GET /documents/download` passaram a resolver contra o banco (`Document.title`/`storage_path`) em vez de varrer `regimentos_estatutos_resolucoes/` — sem isso, documentos enviados pela nova API (armazenados em `data/raw/`) nunca apareceriam nesses endpoints públicos.
- `scripts/create_admin.py` — bootstrap do primeiro `AdminUser`.
- `JWT_SECRET`, `JWT_EXPIRE_MINUTES`, `RAW_DOCS_DIR` em `src/config.py` — mesmo padrão de fallback seguro já usado na Fase 0/1 (se `JWT_SECRET` não estiver definida, gera uma aleatória em memória com aviso no log, em vez de usar um segredo previsível).

### Decisões de Arquitetura

#### D18 — Upload síncrono dentro do request (sem fila assíncrona)

**Decisão:** `POST /admin/documents` roda o pipeline completo (ETL→chunk→embed→index) antes de responder — sem Celery/RQ/background job.

**Justificativa:** já prevista em `PLANO_V2.md` §7.4 para o volume atual do projeto (dezenas de documentos, não milhares). Introduzir uma fila agora seria infraestrutura sem necessidade comprovada. Se o tempo de resposta do upload se tornar um problema real (documentos muito grandes), essa decisão é revisável isoladamente sem afetar o resto da arquitetura.

#### D19 — Falha de ingestão não derruba o upload (HTTP 201 mesmo com `status="failed"`)

**Decisão:** se `process_document` lançar exceção durante um upload ou reindex, `DocumentService` captura, loga, e retorna o documento normalmente (com `status="failed"`, e o erro registrado em `IngestionJob` pela Fase 1).

**Justificativa:** o upload (criar o registro + salvar o arquivo) e a ingestão (processá-lo) são semanticamente operações diferentes — a primeira teve sucesso mesmo se a segunda falhar. Isso também torna o `status` do `Document` a fonte confiável de verdade sobre "este documento está pesquisável?", em vez de depender do código de status HTTP da requisição de upload.

### Bugs encontrados e corrigidos durante a verificação

1. **`DELETE /admin/documents/{id}` falhava com 500** (`IntegrityError: violates foreign key constraint "ingestion_jobs_document_id_fkey"`). Causa: `IngestionJob` tem uma FK para `Document` mas, ao contrário de `DocumentChunk`, não tem uma `relationship(..., cascade="all, delete-orphan")` configurada em `Document` — o ORM não sabia que precisava apagar essas linhas antes de deletar o documento. Corrigido em `src/documents/service.py`: `delete_document()` agora apaga explicitamente as linhas de `IngestionJob` daquele documento antes do `db.delete(document)`. Encontrado ao testar o fluxo de exclusão de ponta a ponta — a primeira tentativa de `DELETE` já tinha removido os vetores do ChromaDB, o arquivo físico e o JSONL antes de falhar no banco, deixando o `Document` "zumbi" (visível no banco, mas sem conteúdo pesquisável) até a correção e uma segunda tentativa.
2. **Diretório vazio `data/raw/{id}/` ficava para trás após a exclusão** — `delete_document()` só apagava o arquivo, não o diretório pai. Corrigido: tenta `rmdir()` do diretório pai após apagar o arquivo (`except OSError: pass` se não estiver vazio ou já não existir).

### Achado de comportamento (não corrigido — registrado para decisão futura)

Ao reindexar um documento (`POST /admin/documents/{id}/reindex`), `etl_and_chunk` (Fase 1) roda a detecção automática de revogação (`revocation_filter.analyze_revocation`, baseada no nome do arquivo) e **sobrescreve** qualquer marcação manual de `revoked`/`revoked_reason` feita via `PATCH` antes do reindex. Percebido ao testar a sequência PATCH (marcar `revoked=true` manualmente) → reindex (voltou para `revoked=false`). Não é necessariamente um bug — pode ser o comportamento correto (a detecção automática deveria ser re-derivada do documento fonte a cada reindex) ou pode ser indesejado (um admin que sabe que uma norma foi revogada por um motivo que o nome do arquivo não capta perderia essa marcação). Fica como decisão de produto para quando o painel administrativo (frontend) existir e esse fluxo for usado de verdade — não bloqueia nada hoje.

### Verificação realizada

Sequência completa contra a API real (Postgres + ChromaDB via Docker): criação do primeiro `AdminUser`; login (sucesso, senha errada → 401, rota sem token → 401); upload multipart de um PDF pequeno já existente no corpus (`ON 02_2016.pdf`, reaproveitado só como payload de teste) → `201`, `status="indexed"`, 31 `DocumentChunk`s, vetores no ChromaDB; pergunta real via `POST /chat/` recuperou o documento de teste como fonte principal (score 0.999) com citação correta; `GET`/`PATCH` confirmados; `POST /reindex` confirmado (`version` 1→2, mesma contagem de chunks, sem duplicar vetores — 1248 = 1217 + 31 antes e depois); `GET /documents/list` e `GET /documents/download` públicos refletindo o documento de teste (inclusive após renomear via PATCH); `DELETE` confirmado limpando ChromaDB, arquivo físico, diretório, JSONL e as três tabelas (`documents`, `document_chunks`, `ingestion_jobs`) — corpus de volta a exatamente 48 documentos e 1217 chunks no BM25/ChromaDB.

### Nota para a escrita do TCC II

O bug do `DELETE` é um bom exemplo para a seção de **Discussão/Limitações**: modelar uma tabela nova (`IngestionJob`, Fase 0) sem uma relação ORM completa é um erro fácil de cometer e fácil de não notar até que o caminho de código que a exercita (exclusão) seja realmente testado — reforça o valor de testes de ponta a ponta reais (não só sanity import) antes de considerar uma fase concluída, e é uma evidência concreta a favor da metodologia adotada neste projeto (verificar cada fase contra o sistema rodando de verdade, não só ler o código).

---

## Fase 3 — Corpo Docente (dados estruturados + CRUD + seed real do CECOMP)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** fechar a decisão **D9** do `EVOLUTION.md` (v1) — professores são dado estruturado, consultados via SQL, não via RAG — entregando a camada de dados e a API administrativa. A integração com o agente de chat (`ProfessorTool`) fica para a Fase 4, que ainda não existe.

### Mudança de escopo em relação ao plano original

O plano inicial desta fase assumia que o vínculo profissional↔curso passaria por `Discipline`/`ProfessorDiscipline` (M2M criadas na Fase 0). O usuário forneceu os dados reais do corpo docente do CECOMP (Colegiado de Engenharia da Computação) — 15 professores, sendo 7 do NDE (Núcleo Docente Estruturante, um deles Coordenador) — com nome, área principal, Lattes, site pessoal e e-mail. Esse dado é **texto livre de área de atuação**, não um vínculo curricular preciso (código de disciplina, período, carga horária), então população de `Discipline`/`ProfessorDiscipline` a partir dele seria inventar estrutura que os dados não têm. Decisão tomada em conjunto com o usuário: `Professor` ganhou `course_id` (afiliação direta ao curso) e os campos que os dados reais exigem (`area`, `lattes_url`, `personal_site_url`, `is_nde`, `nde_role`, `email_secondary`); `Discipline`/`ProfessorDiscipline` permanecem no schema como infraestrutura para quando houver dado curricular preciso (Fase 5, importação do PPC), sem serem populadas agora.

### O que foi feito

- `Professor` estendido em [src/db/models.py](src/db/models.py) com os campos acima + `cascade="all, delete-orphan"` em `Professor.disciplines`/`Discipline.professors` (correção **proativa** — ver lição da Fase 2 abaixo). Nova migration (`46fd59ba203e`), confirmada tocando apenas `professors`.
- `src/professors/` (novo, mesma forma de `src/documents/`): `service.py` (`ProfessorService` completo + `create_discipline`/`list_disciplines`/`assign_discipline` como infraestrutura), `router.py` (`GET /professors?course_id=&area=&name=`, público via `x-api-key`).
- `src/admin/schemas.py`/`router.py` estendidos: CRUD completo de `/admin/professors` (+ `/disciplines`), com `ProfessorOut` trazendo as disciplinas aninhadas na mesma resposta.
- `scripts/seed_professors_engcomp.py` (novo) — semeia os 15 professores reais, idempotente por e-mail. **Diferente do documento de teste da Fase 2, este dado fica no sistema.**

### Lição da Fase 2 aplicada preventivamente

Na Fase 2, `IngestionJob` sem `relationship(cascade=...)` quebrou o `DELETE` de documentos com um `IntegrityError` só descoberto ao testar de ponta a ponta. Desta vez, antes de escrever o serviço, adicionei `cascade="all, delete-orphan"` em `Professor.disciplines`/`Discipline.professors` diretamente no modelo (Escopo item 1) — e o `DELETE /admin/professors/{id}` de teste, com uma associação de disciplina ativa, funcionou de primeira (204, sem erro), confirmando que vale a pena revisar cascades de relacionamento *antes* de escrever `delete_*()`, não depois de um 500 em produção.

### Verificação realizada

Migration aplicada (só `professors` mudou); seed rodado 2x (15 criados na primeira, 0 na segunda — idempotência); `GET /professors` público testado com os 3 filtros (`course_id=1` → 15, `area=Robótica` → Juracy Emanuel, `name=Jadsonlee` → 1 resultado); conferido via SQL que `is_nde=true` bate exatamente com os 7 do NDE e `nde_role='Coordenador'` só em Brauliro Gonçalves Leal; ciclo completo de CRUD com 1 professor de teste (`POST`/`GET`/`PATCH`) + 1 disciplina de teste + associação — repetida a mesma associação para confirmar **upsert idempotente** (200, dados atualizados, sem `IntegrityError`) em vez de descobrir o problema depois; `DELETE` do professor de teste confirmado limpo (cascade); disciplina de teste removida manualmente (não há endpoint de exclusão — decisão deliberada, ver Fora de Escopo). Estado final: exatamente 15 professores reais, 0 disciplinas residuais.

### Nota para a escrita do TCC II

Bom exemplo para a seção de **Metodologia** de como a modelagem evolui em resposta a dado real, não apenas a especificação teórica: o plano original (`PLANO_V2.md`) previa vínculo via `Discipline`, mas o dado real disponível não sustentava essa estrutura sem inventar informação — a decisão de adicionar `course_id`/`area` diretamente em `Professor`, mantendo `Discipline` como infraestrutura para depois, é um exemplo concreto de "modelar o dado que você tem, preparar para o dado que você vai ter" sem overengineering nem submodelagem.

---

## Fase 4 — Orquestração Multi-Tool (function calling nativo)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** a mudança mais profunda do roadmap (`PLANO_V2.md` §10.7) — substituir a decisão binária manual ("buscar ou não", um JSON parseado à mão) por *function calling* nativo da OpenAI, com um registro de Tools extensível. Resolve **D5** (o parsing manual não escalava para múltiplas ferramentas — era literalmente o que bloqueava plugar o `ProfessorTool` da Fase 3) e **D4** (a duplicação quase total entre `run_chat`/`stream_chat`).

### O que foi feito

- `src/agent/tools/rag_tool.py` e `src/agent/tools/professor_tool.py` (novos): wraps finos sobre o pipeline de retrieval (HyDE + busca híbrida + reranking, Fases anteriores) e sobre `ProfessorService` (Fase 3) — **nenhuma lógica de busca foi reimplementada**, só reorganizada atrás de um contrato uniforme (`execute(arguments, context) -> {"summary": str, "sources": list[dict]}`).
- `src/agent/orchestrator.py` (novo): uma única função geradora `run()` — chama o LLM com `tools=[...]`; sem `tool_calls`, a própria resposta já é final (cumprimentos/follow-ups); com `tool_calls` (1 ou mais — a API já suporta paralelismo nativo), executa cada tool, monta as mensagens `role: "tool"`, e faz uma segunda chamada (streaming, com `stream_options={"include_usage": True}` para não perder a contagem de tokens) para sintetizar a resposta final com citação obrigatória.
- `src/chat/service.py` reescrito: `run_chat`/`stream_chat` mantiveram a **assinatura pública exata de antes** — viraram adaptadores finos que consomem o generator do orchestrator. `chat/router.py` **não precisou de nenhuma mudança**.
- `src/chat/schemas.py`: extensão aditiva — `SourceInfo.origin` (`"rag"` | `"professor"`) e `ChatResponse.used_tools`. Contrato antigo continua válido para quem só lê os campos de sempre.
- `API.md` atualizado com os campos novos.

### Decisão de arquitetura: sessão do banco gerenciada dentro do generator, não via `Depends(get_db)`

Diferente do padrão estabelecido nas Fases 2/3 (`Depends(get_db)` no router, `db` passado pra service), aqui a sessão é aberta/fechada **dentro de** `run_chat`/`stream_chat`. Motivo específico: com `StreamingResponse`, o generator (`stream_chat`) só é consumido pelo Starlette **depois** que a função da rota já retornou — uma sessão injetada via `Depends` seria fechada antes do generator (que precisa de `db` para o `ProfessorTool`) terminar de rodar. É um gotcha conhecido de FastAPI + `Depends` + streaming, evitado aqui antes de virar bug — na mesma linha da lição da Fase 2 (cascade de `IngestionJob`) e da correção preventiva da Fase 3 (cascade de `Professor.disciplines`): revisar o ciclo de vida de recursos *antes* de escrever o código, não depois de um erro em produção.

### Por que `src/generation/generator.py` e `src/evaluation/` não foram tocados

`scripts/run_eval.py`/`src/evaluation/ragas_eval.py` chamam `generator.generate_answer()`/`SYSTEM_PROMPT` diretamente, sem passar por `chat/service.py` — é um caminho deliberadamente desacoplado do agente, usado para medir a qualidade "crua" do pipeline de retrieval+geração (baseline RAGAS). Preservar esse módulo intocado nesta fase mantém esse baseline válido para comparações futuras; a lógica de citação/resposta do **agente** agora vive só no `SYSTEM_PROMPT` do orchestrator.

### Achado (não é bug, registrado para calibração futura)

Em uma pergunta de teste combinando NDE ("quem coordena o NDE de Engenharia da Computação?"), o `search_professors` foi corretamente acionado mas não encontrou o professor porque a busca por `name`/`area` não cobre o campo `nde_role` — o agente respondeu honestamente "não encontrei" em vez de inventar. Correto do ponto de vista de segurança contra alucinação, mas é uma lacuna de cobertura da tool (poderia aceitar um parâmetro `nde_role`/`is_nde` para esse tipo de pergunta). Não corrigido agora — funcionalidade nova, não regressão, fica como refinamento futuro da tool.

### Verificação realizada

Sequência completa contra a API real: saudação sem tools (`used_tools=[]`); pergunta normativa (trancamento de matrícula) → `search_normative_documents`, citação correta do Art. 72; pergunta sobre professor (Jadsonlee da Silva Sá) → `search_professors`, dados batendo com o seed da Fase 3; pergunta combinada sobre NDE → **as duas tools na mesma resposta**, com fontes `origin="rag"` e a tentativa de `origin="professor"` (achado acima); streaming (`/chat/stream`) repetido para pergunta normativa e de professor — 59 eventos `token` reais (token-a-token), evento `done` com `sources`/`used_tools` idênticos ao endpoint não-streaming para a mesma pergunta; follow-up com histórico (pergunta normativa → "e qual o prazo?") — novo `tool_call` disparado com o contexto da conversa, resposta honesta quando a informação não estava no contexto recuperado; `data/logs/queries.jsonl` confirmado registrando `used_search` corretamente para todos os casos acima.

### Nota para a escrita do TCC II

Esta fase é o núcleo da seção de **Arquitetura do Agente** do TCC II — é a evidência mais direta de que "Advanced RAG" evoluiu para algo mais próximo de "Modular RAG" (na taxonomia do TCC I, seção 2.1: Naive → Advanced → Modular), com um LLM decidindo autonomamente entre múltiplas fontes de conhecimento (não-paramétrica via RAG, estruturada via SQL) em vez de um pipeline fixo de uma única fonte. O teste da pergunta combinada sobre NDE é um bom exemplo para ilustrar isso na monografia — mostra o agente escolhendo E combinando fontes heterogêneas numa única resposta.

---

## Fase 5a — PPC Real: Conteúdo Textual (RAG) + Matriz Curricular (Estruturada)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** ingerir o PPC real (144 páginas) e o corpo docente/matriz curricular oficiais da UNIVASF, resolvendo um problema de arquitetura que o v1 não tinha enfrentado: um documento que mistura texto normativo estruturado por artigo com prosa narrativa.

### Mudança de plano durante a discussão

O plano inicial dividia manualmente o PPC em 6 PDFs (4 normativos + 2 de prosa) antes do upload. O usuário questionou isso corretamente: "existem muitas páginas que não se enquadram nesse formato de chunking, e o Manual do Aluno também, além de no futuro surgir a necessidade de adicionar mais documentos gerais pela plataforma admin" — ou seja, a divisão manual resolveria o PPC de hoje mas não generaliza para nenhum documento futuro. Redesenhei para que a decisão de chunking (legal vs. heading) aconteça **por bloco, dentro de `legal_chunker.chunk_document()`**, não por documento inteiro.

### O que foi feito

- `src/chunking/heading_chunker.py` (novo): `split_prose_block()` — divide por headings Markdown, sub-divide por parágrafo se uma seção exceder `MAX_CHUNK_TOKENS`.
- `src/chunking/legal_chunker.py::chunk_document()`: quando um bloco não começa com "Art. X", chama `heading_chunker.split_prose_block()` em vez de criar um único chunk gigante. `src/ingestion/service.py` não mudou — a decisão fica encapsulada onde já fazia sentido.
- **Upload do PPC completo (144 páginas) como um único documento**, via `POST /admin/documents` (Fase 2, zero código novo) — `course_id=1` (ENGCOMP), `knowledge_base_slug=regulamentos`. Resultado: 375 chunks — 128 com `article_id` (Cap. 6, Documentos Normativos: Regulamento do Colegiado, Curricularização de Extensão, Estágio, TCC) e 247 sem (Identificação, Estrutura Curricular narrativa, Ementário, Infraestrutura), roteados automaticamente para o chunker certo.
- `scripts/seed_disciplines_engcomp.py` (novo): 81 disciplinas da matriz curricular real (63 dos 10 períodos + 18 optativas), com sigla, período, carga horária e `prerequisites_text` (texto livre) — via `ProfessorService.create_discipline` (Fase 3).
- `Professor.degree` (novo campo, migration) + `Discipline.prerequisites_text` (novo campo, mesma migration): `degree` sincronizado nos 15 professores já existentes (12 Doutor(a) + 3 Mestre, confirma exatamente a Tabela 5.3 do PPC) sem recriar registros.

### Achado real durante a verificação — e uma tool nova não planejada originalmente

Testando "em que período fica Compiladores e quais os pré-requisitos?" **usando só RAG**, o agente respondeu período correto (7º) mas **pré-requisitos errados**: "Teoria da Computação e Estruturas de Dados" em vez de "AED, LFA e OAC" (a matriz curricular real). Isso é exatamente o padrão que já resolvi para professores (D9): fato exato não deveria depender de recall de embedding sobre prosa. Adicionei `src/agent/tools/discipline_tool.py` (`search_disciplines`, wrap fino sobre `list_disciplines` — já existia desde a Fase 3, zero lógica nova) e registrei no orchestrator junto com `RagTool`/`ProfessorTool`. Reperguntado com a tool disponível: resposta correta ("AED, LFA e OAC"), **e mais rápida** (3,9s vs. ~17s do caminho via RAG+HyDE+rerank) — o SQL exato venceu em precisão e em latência.

`SourceInfo.origin` ganhou um terceiro valor (`"discipline"`, aditivo) e `_build_source_infos` em `chat/service.py` foi estendido para esse caso.

### Verificação realizada

Teste isolado de roteamento por bloco com texto sintético misto (confirmado antes de tocar no PPC real); upload do PPC completo confirmado com a mistura esperada de chunks (128 legal / 247 heading); seed de disciplinas rodado 2x (81 criadas, depois 0 — idempotência); `degree` sincronizado e conferido via SQL (12 Doutor(a) + 3 Mestre = 15, bate com a Tabela 5.3); perguntas reais via `/chat/` sobre conteúdo de prosa do PPC (duração/carga horária do curso — resposta correta, citando `origin="rag"` com `hierarchy="4.1 Organização do Currículo"`) e sobre o Regulamento do TCC do PPC (banca examinadora — `Art. 24`, `source="PPC - Engenharia de Computação"`, distinguível de outras normas do corpus); achado e corrigido o caso de pré-requisitos incorretos, com nova tool testada e confirmando resposta exata.

**Nota operacional:** durante a verificação, duas chamadas consecutivas de chat com timeout de cliente (curl) deixaram requisições órfãs concorrendo por CPU no processo do servidor (o `CrossEncoder` do reranker é compartilhado e não é thread-safe para uso concorrente pesado), causando lentidão aparente de "travamento". Reiniciar o processo resolveu — não é um bug do código desta fase, mas um lembrete de que testes de carga/concorrência do reranker ficam como item para investigar se o volume de uso real crescer.

### Nota para a escrita do TCC II

O achado do `DisciplineTool` é talvez o exemplo mais didático do projeto inteiro para a seção de **Resultados/Discussão**: uma alucinação parcial real, capturada ao vivo durante o desenvolvimento, com causa raiz identificada (fato exato dependendo de RAG) e correção que já tinha precedente arquitetural (D9) — e a correção melhorou precisão E latência simultaneamente. É uma demonstração concreta de por que a arquitetura "RAG para prosa, SQL para fatos exatos" (`PLANO_V2.md`, Seção 5) não é só uma preferência teórica.

---

## Fase 5b — Calendário Acadêmico (dado estruturado + Tool)

**Status:** ✅ Concluída e verificada — 2026-07-03

**Objetivo:** tratar o Calendário Acadêmico de Graduação como dado estruturado consultado via Tool, não como texto de RAG — mesma lógica de professores (D9) e da matriz curricular (Fase 5a). O usuário anexou o PDF real do Calendário 2026 já na discussão da Fase 5, o que puxou para agora a Fase 7 do roadmap original (`PLANO_V2.md`), que estava marcada como condicional "só se houver demanda validada" — a demanda passou a existir assim que o dado real ficou disponível.

### O que foi feito

- `AcademicEvent` (novo model, migration): `course_id` nullable (`None` = vale para todos os cursos, é o caso de 100% dos eventos deste calendário — não é específico de Engenharia de Computação), `title`, `start_date`, `end_date` nullable (evento de um dia), `category`, `legal_reference` (preserva a mesma rastreabilidade normativa já aplicada ao RAG — vários eventos do calendário citam artigos/resoluções específicas), `campus` nullable (`None` = todos os campi; alguns feriados municipais valem só para um campus), `academic_period` (ex.: `"2026.1"`).
- `src/calendar_events/` (novo módulo, mesmo formato de `src/professors/`): `service.py` (`create_event`, `list_events` com filtros por `course_id`/`category`/`academic_period`/intervalo de datas, `get_event`, `update_event`, `delete_event`); `router.py` — único endpoint público `GET /academic-events`, protegido por `x-api-key` (mesmo padrão de `/professors`).
- `src/admin/schemas.py`/`router.py` estendidos com CRUD completo (`POST`/`GET`/`GET {id}`/`PATCH`/`DELETE /admin/academic-events`), mesmo formato do CRUD de professores/disciplinas da Fase 3.
- `src/agent/tools/calendar_tool.py` (novo, `search_academic_calendar`): wrap fino sobre `calendar_events.service.list_events` — zero lógica de consulta nova, mesmo contrato uniforme (`execute(arguments, context) -> {"summary", "sources"}`) das outras três tools. Registrado em `orchestrator.py` (`TOOLS`/`TOOL_SCHEMAS`/`STATUS_MESSAGES`) e no `SYSTEM_PROMPT`, com uma instrução explícita para não deixar `search_normative_documents` responder por prazos exatos.
- `SourceInfo.origin` ganhou um quarto valor (`"calendar"`, aditivo) e `_build_source_infos` em `chat/service.py` foi estendido para esse caso.
- `scripts/seed_calendar_2026.py` (novo): 121 eventos reais extraídos do PDF completo do Calendário Acadêmico 2026 (janeiro/2026 a março/2027) — início/fim de período letivo, prazos de matrícula/rematrícula/trancamento, feriados nacionais/estaduais/municipais por campus (JUA, PNZ, PAV, SAL, SBF, SRN), colação de grau, exames finais e demais prazos administrativos. Idempotente por `(title, start_date)`.

### Verificação realizada

Migration aplicada isoladamente (só `academic_events`, autogenerate confirmou); seed rodado 2x — 121 criados na primeira execução, 0 na segunda (idempotência); `GET /academic-events?academic_period=2026.1` (público, `x-api-key`) retornando os eventos corretos; ciclo CRUD completo em `/admin/academic-events` com um evento de teste (criado com um `AdminUser` efêmero, removido ao final junto com o admin de teste); pergunta real via `POST /chat/` — **"Quando é o período de trancamento de matrícula em 2026.1?"** → aciona `search_academic_calendar`, responde "6 a 10 de abril de 2026" (data exata do PDF real, `origin="calendar"`) em vez de arriscar uma resposta aproximada via RAG; segunda pergunta ("Quando começam as aulas do período 2026.2?") → resposta correta (10/08/2026) filtrando por categoria de período letivo; regressão de saudação sem tools confirmada (`used_tools=[]`).

### Nota para a escrita do TCC II

Junto com o achado do `DisciplineTool` na Fase 5a, esta fase fecha o terceiro (e último, no escopo atual) exemplo empírico do mesmo padrão arquitetural: fato exato → SQL/Tool, prosa → RAG. Vale como uma tabela comparativa na seção de Resultados — três domínios de dado (corpo docente, matriz curricular, calendário acadêmico), mesma decisão de design aplicada três vezes, cada uma validada contra uma pergunta real que uma busca semântica pura responderia com risco de erro.

---

## Itens Pendentes / Próximos Passos

> Atualizar ao final de cada fase.

- [ ] Fase 6 (escopo por curso): expor `course_id` em `ChatRequest`, filtro `$or` no retrieval, `GET /courses`.
- [ ] Refinar `ProfessorTool` para aceitar `nde_role`/`is_nde` como parâmetro de busca (achado da Fase 4).
- [ ] Decidir o comportamento de `revoked` em reindex de documentos (achado da Fase 2) quando o painel admin existir.
- [ ] Ativar `score_threshold` no reranker (D3 do diagnóstico) — pendente de calibração empírica contra a distribuição real de scores do `BAAI/bge-reranker-v2-m3` (achado da Fase 4: não é garantidamente 0–1 como o Cohere Rerank que a metodologia do TCC1 assume).
- [ ] Considerar execução concorrente de tools quando o agente pede mais de uma na mesma resposta (hoje é sequencial — ver Fase 4, Fora de Escopo). O achado operacional da Fase 5a (contenção do reranker sob requisições concorrentes/órfãs) reforça que isso vale investigar se o volume de uso crescer.
