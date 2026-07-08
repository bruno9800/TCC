# Planejamento Arquitetural — Segunda Versão do Agente RAG UNIVASF

> Documento técnico de arquitetura e planejamento para a evolução do backend do assistente normativo da UNIVASF. Elaborado a partir da leitura completa do código-fonte (`src/`), da documentação viva do projeto (`README.md`, `API.md`, `EVOLUTION.md`, `rules.md`), do `docker-compose.yml`, e do corpus indexado (`data/chunks/`, 1.216 chunks em 46 documentos). O TCC1 (anteprojeto) foi usado apenas para entender objetivos e restrições — não é avaliado aqui.

---

## 1. Entendimento do Sistema Atual

### 1.1 Visão geral do fluxo

O sistema é uma API **FastAPI** stateless que implementa um pipeline **Advanced RAG** de dois estágios sobre um corpus fixo de 48 PDFs normativos da UNIVASF (Estatuto, Regimento Geral, Resoluções PROEN/PROEX/PRPPGI). Não há frontend neste escopo (React consome a API separadamente) e não há banco de dados relacional — toda persistência é feita em arquivos (JSONL) e no ChromaDB (vetores + metadados).

O fluxo de uma requisição de chat (`POST /chat/` ou `POST /chat/stream`) é:

1. **Decisão do agente** ([src/chat/service.py](src/chat/service.py)) — uma primeira chamada ao LLM (`gpt-4o`, temperatura 0.0) recebe o `SYSTEM_PROMPT` + histórico + mensagem, e retorna um JSON manualmente parseado: `{"needs_search": bool, "search_query"|"direct_response": str}`. Isso decide entre resposta direta (saudações, follow-ups sobre a própria conversa) ou acionar o pipeline RAG.
2. **HyDE** ([src/retrieval/hyde.py](src/retrieval/hyde.py)) — se precisa buscar, `gpt-4o-mini` gera um parágrafo hipotético no "dialeto normativo" (estilo artigo de lei) para a query. Esse texto é embedado e o vetor resultante substitui o embedding da query original **apenas na busca densa** (o BM25 continua usando a query crua).
3. **Busca híbrida** ([src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py)) — `HybridSearchEngine` combina busca densa via ChromaDB/HNSW (`INITIAL_TOP_K=20`, filtro hard `status:"vigente"` quando `filter_revoked=True`) com BM25 (`rank_bm25`, índice construído em memória a partir de **todos** os chunks carregados do disco), fundidos por Reciprocal Rank Fusion (pesos 0.7 denso / 0.3 esparso, k=60).
4. **Reranking** ([src/retrieval/reranker.py](src/retrieval/reranker.py)) — Cross-Encoder `BAAI/bge-reranker-v2-m3` (`sentence-transformers`, carregado uma vez como singleton) reordena os candidatos e corta em `top_k` (padrão 5). O parâmetro `score_threshold` existe na função mas **não é usado** no `service.py`.
5. **Geração** ([src/generation/generator.py](src/generation/generator.py) / lógica duplicada em `service.py`) — `build_context()` formata os chunks selecionados com cabeçalho de metadados (fonte, categoria, artigo, hierarquia) e injeta no `ANSWER_PROMPT` (regras de citação obrigatória, restrição a responder apenas com base no contexto, formato conciso). `gpt-4o`, temperatura 0.1.
6. **Resposta** — `ChatResponse` com `answer`, `sources` (deduplicadas por documento, com `download_url` já pronto), `tokens`, `used_search`. A versão `/stream` emite os mesmos dados como eventos SSE (`status`, `token`, `done`, `error`).
7. **Log** ([src/logs/query_logger.py](src/logs/query_logger.py)) — cada interação é apensada a `data/logs/queries.jsonl`, exposta via `GET /logs/queries` e `GET /logs/stats`.

### 1.2 Pipeline de ingestão (offline, hoje 100% manual)

Não existe API de ingestão. O fluxo é executado via dois scripts CLI:

- `scripts/run_etl.py` — varre `regimentos_estatutos_resolucoes/**/*.pdf`, converte cada PDF para Markdown via `pymupdf4llm` ([src/etl/pdf_converter.py](src/etl/pdf_converter.py)), classifica a categoria pelo caminho do diretório (PROEN/PROEX/PRPPGI/raiz), detecta revogação por regex no nome do arquivo ([src/etl/revocation_filter.py](src/etl/revocation_filter.py)), aplica o chunking semântico-hierárquico ([src/chunking/legal_chunker.py](src/chunking/legal_chunker.py)) e salva um `.jsonl` por documento em `data/chunks/`.
- `scripts/run_indexing.py` — carrega **todos** os JSONL de `data/chunks/` (`load_all_chunks()`), gera embeddings (`text-embedding-3-large`) e faz `upsert` em uma única collection ChromaDB (`univasf_normas`).

O chunking merece destaque: cada chunk corresponde a um Artigo completo (caput + parágrafos + incisos), com fallback recursivo para artigos longos que excedem `MAX_CHUNK_TOKENS` (512) — nesse caso, os sub-chunks herdam o caput como prefixo de contexto. A hierarquia (Título > Capítulo > Seção) é extraída por posição no texto e serializada como string plana em `metadata.hierarchy` (ChromaDB só aceita metadados escalares). Isso é uma implementação fiel e funcional do que o TCC1 propõe teoricamente (LC 95/1998).

### 1.3 Responsabilidades por módulo

| Módulo | Responsabilidade | Estado |
|---|---|---|
| `src/etl/` | PDF → Markdown, classificação por pasta, detecção de revogação | Funcional, acoplado a `pymupdf4llm` |
| `src/chunking/` | Segmentação por artigo + herança de contexto | Funcional, específico para texto legal estruturado |
| `src/indexing/` | Embeddings OpenAI + persistência ChromaDB | Funcional, uma única collection |
| `src/retrieval/` | Busca híbrida (dense+BM25+RRF) + HyDE + reranking | Funcional, BM25 carregado 1x no processo |
| `src/generation/` | Prompt de resposta + formatação de contexto | Funcional, mas duplicado dentro de `chat/service.py` |
| `src/chat/` | Orquestração do agente (decide buscar ou não) + streaming | Funcional, decisão binária via JSON manual |
| `src/documents/` | Listagem, busca semântica leve (sem HyDE), download de PDF | Funcional, acesso direto ao filesystem |
| `src/logs/` | Log de interações em JSONL + agregações simples | Funcional, sem persistência estruturada |
| `src/evaluation/` | RAGAS offline sobre golden dataset de 15 perguntas | Funcional, desacoplado da API |

### 1.4 Integração entre componentes

Tudo roda em um único processo Python. Os "singletons" (`HybridSearchEngine`, `CrossEncoder`, `OpenAI client`) são inicializados lazy e mantidos em memória entre requisições — isso evita custo de reload, mas cria um acoplamento importante: **o estado do índice BM25 e da lista de chunks é um snapshot carregado na inicialização do processo**, não uma consulta ao vetorstore em tempo real (diferente da busca densa, que sempre consulta o ChromaDB diretamente).

Não há autenticação de usuário — apenas uma API key estática (`x-api-key`) validada em `src/auth.py`, aplicada a todos os routers (`chat`, `documents`, `logs`) em [src/main.py](src/main.py). Se `TCC_API_KEY` não estiver definida, o acesso é liberado (modo dev inseguro, documentado no próprio código).

### 1.5 Pontos fortes

1. **Separação de responsabilidades limpa** — cada estágio do pipeline (ETL, chunking, indexação, retrieval, geração) é um módulo isolado, testável independentemente. Isso facilita muito a evolução incremental.
2. **Chunking juridicamente informado** — a decisão de segmentar por dispositivo legal (não por tamanho fixo) é sofisticada e corretamente implementada, incluindo o caso de borda de artigos longos.
3. **Recuperação em funil bem desenhada** — busca híbrida para maximizar recall, cross-encoder para maximizar precisão. É literalmente o padrão "Advanced RAG" descrito na fundamentação teórica.
4. **HyDE já implementado** — vai além do que o TCC1 (anteprojeto) propunha originalmente; mostra que a arquitetura já evoluiu organicamente além do papel.
5. **Filtro de vigência (hard filter) antes da busca** — mitiga o risco mais grave do domínio (citar norma revogada como se fosse válida).
6. **API desacoplada do frontend**, com streaming SSE já funcional — bom ponto de partida para evoluir a experiência conversacional sem reescrever a camada de transporte.
7. **Documentação viva** (`EVOLUTION.md` com decisões justificadas, `API.md` com contratos) — raro em projetos deste porte, e um ativo real para este planejamento (a decisão D9 sobre professores, por exemplo, já antecipa parte do que este documento formaliza).

### 1.6 Limitações (descritas aqui apenas para registro; diagnóstico técnico na Seção 2)

- Uma única fonte de conhecimento (normas institucionais), sem banco relacional, sem API de ingestão, sem conceito de curso, sem dados estruturados de professores.

---

## 2. Diagnóstico

### 2.1 O problema relatado, verificado nos dados

Antes de aceitar a premissa "preciso de múltiplas bases de conhecimento", verifiquei o corpus real (`data/chunks/`, 46 arquivos). Resultado: **PPC e documentação de estágio já estão indexados** na mesma collection — `Resolucao-PPC-23.jsonl`, `Cartilha_Lei_do_Estágio.jsonl`, `Lei_11.788_2008.jsonl`, `IN_213_2019_.jsonl`, `ON_02_2016.jsonl`, `Resolução_09_2022_...estagios.jsonl` já existem e são recuperáveis pelo pipeline atual. Ou seja: **o problema não é fragmentação de índices** (isso já é um único pool de busca, o que é bom). O que **de fato não existe em lugar nenhum** do corpus é:

- **Manual do Aluno** — não há esse documento em `regimentos_estatutos_resolucoes/`.
- **Calendário Acadêmico** — não há esse documento em lugar nenhum.
- **Corpo Docente** — não existe nenhuma fonte de dados sobre professores.

Isso muda a leitura do problema: a limitação real é **cobertura de conteúdo insuficiente + falta de infraestrutura para adicionar novos documentos/tipos de dado**, não uma falha estrutural da estratégia de recuperação em si. Isso é uma boa notícia arquitetural — o pipeline de recuperação (híbrido + HyDE + rerank) não precisa ser refeito; precisa ser **alimentado com mais conteúdo** e **estendido para tipos de dado que não são texto normativo**.

### 2.2 Gargalos e riscos técnicos concretos

| # | Achado | Evidência | Risco |
|---|---|---|---|
| D1 | **Índice BM25 é um snapshot congelado na inicialização do processo.** `HybridSearchEngine.__init__` chama `load_all_chunks()` e constrói o BM25 uma única vez. A busca densa consulta o ChromaDB ao vivo, mas a busca esparsa não. | [src/retrieval/hybrid_search.py:48-56](src/retrieval/hybrid_search.py) | Depois de qualquer nova indexação, o componente esparso da busca híbrida fica desatualizado até o processo reiniciar. Isso **bloqueia diretamente** um dos requisitos da Seção 7 do escopo pedido ("disponibilização para consultas" logo após o upload). |
| D2 | **Reindexação não remove chunks órfãos.** `index_chunks()` só faz `upsert`; se um documento for re-chunkeado com limites de artigo diferentes, os IDs antigos (`source__article__index`) que não existirem mais no novo conjunto continuam no ChromaDB para sempre. | [src/indexing/vector_store.py:121-185](src/indexing/vector_store.py) | Vetores fantasmas retornáveis pela busca, sem rastreabilidade — risco de citar um trecho de uma versão desatualizada do artigo. |
| D3 | **`score_threshold` do reranker existe mas não é usado.** O TCC1 descreve explicitamente um corte de relevância (σ > 0.4) como mecanismo de segurança contra ruído; o código implementa o parâmetro em `rerank()` mas `chat/service.py` chama sem passá-lo. | [src/retrieval/reranker.py:34-39](src/retrieval/reranker.py) + [src/chat/service.py:229](src/chat/service.py) | O sistema sempre retorna top-5 por *rank*, mesmo quando nenhum candidato é de fato relevante — o mecanismo de "não invente se não houver contexto bom" fica mais fraco do que a metodologia original prevê. |
| D4 | **Duplicação de lógica entre `run_chat` e `stream_chat`.** Parsing da decisão, montagem de contexto e montagem de `sources` estão copiados quase literalmente nas duas funções. | [src/chat/service.py:118-463](src/chat/service.py) | Qualquer mudança na orquestração (ex.: adicionar uma nova Tool) precisa ser replicada em dois lugares — risco real de divergência já com apenas duas variantes; piora exponencialmente ao adicionar tools. |
| D5 | **Decisão "buscar ou não" é um parsing manual de JSON, não function calling nativo.** Frágil por natureza (o código já tem um patch ad hoc para strings envolvidas em ```` ``` ````) e **não escala** para múltiplas ferramentas (normas, professores, calendário) sem reescrever a lógica de decisão do zero. | [src/chat/service.py:168-187](src/chat/service.py) | É o principal bloqueador arquitetural para dar ao agente acesso a mais de uma fonte de informação — que é exatamente o problema central que motivou este planejamento. |
| D6 | **Zero banco de dados relacional.** Não há registro de documentos (CRUD), versão, curso, categoria como entidades — apenas arquivos e metadados soltos no vetor. | Ausência confirmada em `src/` e `pyproject.toml` (sem driver de banco algum) | Bloqueia diretamente o painel administrativo pedido na Seção 7 (upload, versionamento, curadoria) e a modelagem de professores como dado estruturado. |
| D7 | **Ingestão é 100% CLI, sobre o corpus inteiro.** Não há endpoint para adicionar/atualizar um único documento; `run_etl.py`/`run_indexing.py` reprocessam tudo. | [scripts/run_etl.py](scripts/run_etl.py), [scripts/run_indexing.py](scripts/run_indexing.py) | Inviável para um fluxo administrativo real de "sobe um documento novo e ele fica disponível". |
| D8 | **Autenticação é uma única chave estática compartilhada**, sem papéis (aluno vs. admin). | [src/auth.py](src/auth.py) | Aceitável para o MVP público (decisão D6 do `EVOLUTION.md`), mas **insuficiente** no momento em que o mesmo mecanismo precisar proteger upload/exclusão de documentos — qualquer vazamento da chave pública viraria acesso de escrita total. |
| D9 | **Sem conceito de curso/escopo.** Toda a base é tratada como universal; não há metadado que distinga "vale para todos os cursos" de "específico de Engenharia de Computação". | `ChunkMetadata` em [src/chunking/legal_chunker.py:58-69](src/chunking/legal_chunker.py) não tem campo de curso | Bloqueia a ideia (correta) do usuário de selecionar curso antes da conversa — hoje não há onde armazenar essa dimensão. |
| D10 | **CORS `allow_origins=["*"]` com `allow_credentials=True`.** Já sinalizado no próprio código como inseguro para produção. | [src/main.py:23-29](src/main.py) | Baixo risco hoje (sem cookies/sessão), mas deve ser revisitado antes de existir um painel admin autenticado por cookie/JWT. |
| D11 | **Logs em JSONL plano, sem atribuição de usuário/sessão/curso.** Suficiente para o TCC1, mas não dá para filtrar por curso ou por documento de forma eficiente em escala. | [src/logs/query_logger.py](src/logs/query_logger.py) | Não bloqueia nada agora; vira limitação apenas quando o painel administrativo precisar de analytics agregadas por curso/documento. |

### 2.3 Riscos futuros (se a arquitetura não evoluir)

- Se novas categorias de documento forem adicionadas sem metadado de curso/tipo, a única forma de filtrar por curso no futuro será reprocessar o corpus inteiro — dívida técnica cara.
- Se professores forem indexados como texto livre no mesmo pipeline de chunking legal (ao invés de dado estruturado), qualquer pergunta factual ("qual o e-mail do professor X") passa a depender de recall de embedding — risco real de alucinação em dados que deveriam ter precisão de 100%.
- Se a decisão do agente continuar sendo JSON manual, cada nova Tool multiplica a complexidade do parser e a chance de falha silenciosa (fallback genérico "busca por segurança").

---

## 3. Estratégia de Evolução

Princípio geral: **o pipeline de recuperação (ETL → chunking → indexação → busca híbrida → HyDE → rerank → geração) já funciona e resolve bem o problema para o qual foi desenhado — texto normativo estruturado.** Ele não precisa ser reescrito. O que falta é (a) uma camada de dados relacional para tudo que é administrativo/estruturado, (b) uma camada de orquestração multi-tool para o agente escolher entre "buscar normas" e "consultar dado estruturado", e (c) mais conteúdo real coberto pela base já existente.

| Componente | Decisão | Justificativa |
|---|---|---|
| `src/etl/`, `src/chunking/legal_chunker.py` | **Mantém, sem alterações estruturais** | Já implementa corretamente a estratégia validada teoricamente (LC 95/1998); é reaproveitável para qualquer novo documento normativo (Manual do Aluno, se for estruturado por artigos; PPC; Estágio). Adiciona-se apenas um chunker alternativo (heading-based) para documentos que não são artigos legais (ver Seção 6.2). |
| `src/indexing/vector_store.py` | **Estende metadados, mantém arquitetura (1 collection)** | Adiciona `kb_slug` e `course_id` ao `flat_meta`; não cria novas collections (justificativa na Seção 6.1). |
| `src/retrieval/hybrid_search.py`, `hyde.py`, `reranker.py` | **Mantém a lógica, corrige D1 e D3** | Corrige o bug de staleness do BM25 (recarregar após ingestão) e passa a usar `score_threshold` no rerank. Nenhuma reescrita de algoritmo. |
| `src/generation/generator.py` | **Mantém `build_context`/prompt; remove a duplicação com `chat/service.py`** | A lógica de formatação de contexto e citação já está correta — o problema é estar duplicada, não estar errada. |
| `src/chat/service.py` | **Evolui de forma mais profunda: decisão binária → orquestração multi-tool com function calling nativo** | É o único componente que realmente precisa de uma reescrita direcionada, porque a limitação (D5) é estrutural, não incremental. |
| `src/documents/router.py` | **Mantém `download`/`search`; `list` passa a ler do banco (Document) em vez do filesystem** | Preserva o contrato de API já documentado; troca só a fonte de dados por trás. |
| `src/logs/` | **Mantém JSONL no MVP da v2; migração para tabela é opcional/Fase futura** | Não há necessidade comprovada de queries agregadas complexas ainda — evita introduzir escopo não solicitado. |
| — (novo) | **Introduz camada de banco relacional (PostgreSQL + SQLAlchemy + Alembic)** | Única peça de infraestrutura genuinamente nova — necessária para CRUD de documentos, cursos, professores e para o painel administrativo. Detalhada na Seção 7. |
| — (novo) | **Introduz camada de Tools/orquestração (`src/agent/`)** | Formaliza o que hoje é ad hoc em `chat/service.py`; é o mecanismo que permite ao agente combinar RAG com dados estruturados. |

Essa tabela é, na prática, o critério de "o que evolui vs. o que fica" pedido — qualquer item fora dela (ex.: reescrever o chunker, trocar ChromaDB, trocar o modelo de embedding) foi deliberadamente descartado por não resolver o problema relatado e por violar o princípio de evolução incremental.

---

## 4. Arquitetura da Segunda Versão

### 4.1 Visão modular

```
src/
├── etl/                    [inalterado]
├── chunking/
│   ├── legal_chunker.py    [inalterado]
│   └── heading_chunker.py  [NOVO] fallback para docs não-legais (Manual do Aluno, se necessário)
├── indexing/
│   └── vector_store.py     [estendido: kb_slug, course_id nos metadados]
├── retrieval/
│   ├── hybrid_search.py    [corrigido: reload do índice BM25]
│   ├── hyde.py              [inalterado]
│   └── reranker.py          [corrigido: score_threshold aplicado]
├── generation/
│   └── generator.py         [inalterado — vira a única fonte de build_context/prompt]
├── agent/                   [NOVO — orquestração multi-tool]
│   ├── orchestrator.py      decide e executa tools via function calling nativo
│   ├── tools/
│   │   ├── rag_tool.py       wraps hybrid_search + hyde + rerank (normas)
│   │   └── professor_tool.py consulta estruturada (SQL) sobre professores
│   └── prompts.py            system prompts centralizados (elimina duplicação)
├── chat/
│   ├── router.py             [inalterado na forma — chama orchestrator em vez de run_chat direto]
│   ├── service.py            [simplificado: delega ao orchestrator]
│   └── schemas.py            [estendido: course_id opcional no ChatRequest]
├── documents/
│   └── router.py              [list passa a consultar Document via DB]
├── courses/          [NOVO] CRUD de cursos + GET público /courses
├── professors/       [NOVO] CRUD de professores/disciplinas + GET público /professors
├── knowledge_bases/  [NOVO] taxonomia de bases de conhecimento (mhoritariamente seed/admin)
├── ingestion/         [NOVO] serviço reaproveitável de ETL→chunk→embed→index,
│                              usado tanto pelos scripts CLI quanto pela API admin
├── admin/             [NOVO] auth de administrador (AdminUser) + endpoints /admin/*
├── db/                [NOVO] engine SQLAlchemy, sessão, modelos, migrations (Alembic)
├── logs/               [inalterado]
├── evaluation/          [inalterado]
├── auth.py             [inalterado — continua protegendo as rotas públicas]
├── config.py            [estendido — novas envs: DATABASE_URL, JWT_SECRET, etc.]
└── main.py               [inclui os novos routers]
```

### 4.2 Fluxo do agente na v2

```
Usuário → POST /chat (message, history, course_id?)
              │
              ▼
   AgentOrchestrator.run(message, history, course_id)
              │
              ▼
   Chamada ao LLM com tools=[RagTool.schema, ProfessorTool.schema]
   (function calling nativo — substitui o parsing manual de JSON)
              │
   ┌──────────┴──────────┐
   │ 0 tools chamadas      │ → resposta direta (saudação, follow-up)
   │ 1+ tools chamadas     │
   └──────────┬──────────┘
              ▼
   Executa tools em paralelo (o SDK da OpenAI suporta parallel tool calls):
     RagTool.search(query, course_id)      → hybrid_search + HyDE + rerank (como hoje)
     ProfessorTool.search(nome/disciplina)  → SELECT estruturado via SQLAlchemy
              │
              ▼
   Resultados das tools voltam como "tool" messages → segunda chamada ao LLM
   para síntese final (com o mesmo ANSWER_PROMPT de citação obrigatória)
              │
              ▼
   ChatResponse (ou stream SSE) — sources agora podem vir de RAG e/ou de dados estruturados
```

Streaming: a resposta de síntese final (depois das tools resolvidas) é a única etapa que precisa ser `stream=True`; a decisão de quais tools chamar não precisa (e não deve) ser transmitida em streaming — ela já é rápida e é internamente um passo discreto, igual hoje.

### 4.3 Curso como dimensão transversal, não como partição rígida

`course_id` é **opcional e nullable** em toda a cadeia (`Document`, `Professor`, e no payload de `/chat`):

- `course_id = NULL` em um documento → aplica-se a todos os cursos (Estatuto, Regimento Geral — como já é hoje).
- `course_id = <curso>` → documento/professor específico daquele curso (PPC, disciplinas).
- Uma requisição de chat sem `course_id` continua funcionando exatamente como a v1 (retrocompatibilidade real, não apenas teórica).
- Quando `course_id` é enviado, o filtro de retrieval passa a ser `status:"vigente" AND (course_id == X OR course_id IS NULL)` — ChromaDB suporta esse `$or` nativamente no `where`.

Essa é uma alternativa deliberadamente mais simples do que "cada curso tem sua própria base": evita duplicar documentos institucionais por curso e evita N collections. Ver justificativa completa e comparação de alternativas na Seção 6.1.

---

## 5. Organização da Memória do Agente

| Tipo de informação | Onde fica | Justificativa |
|---|---|---|
| Texto normativo extenso (Estatuto, Regimento, Resoluções, Estágio, PPC narrativo) | **RAG** (ChromaDB, chunking legal existente) | Conteúdo interpretativo e longo; precisa de citação textual e de recall semântico. Já funciona — apenas recebe mais documentos e metadados de curso/kb. |
| Fatos estruturados de professores (nome, e-mail, departamento, disciplinas) | **Banco de dados relacional + Tool** | Consulta exata, tolerância zero a alucinação, precisa de filtros compostos (curso + disciplina). RAG adicionaria latência (embedding+busca+rerank) e risco de erro semântico onde uma `WHERE` simples resolve com 100% de precisão. Valida a intuição do usuário e a decisão D9 já registrada em `EVOLUTION.md`. |
| Calendário acadêmico (datas, prazos, eventos) | **Banco de dados relacional + Tool** (não RAG) | Dados tabulares/temporais. Perguntas como "até quando vai a matrícula?" exigem resposta exata por intervalo de data — RAG tende a aproximar semanticamente e pode citar a data errada com confiança. SQL `WHERE date BETWEEN` é trivial e determinístico. |
| Grade curricular/matriz de disciplinas por período (se extraída do PPC) | **Banco de dados relacional** (view estruturada, derivada do PPC uma vez) + o **texto integral do PPC continua no RAG** | "Quais disciplinas do 5º período?" precisa de resposta exata; a ementa/justificativa de cada disciplina continua sendo prosa, consultada via RAG. |
| Metadados administrativos de documentos (categoria, curso, status de vigência, versão, quem subiu, quando) | **Banco de dados relacional** (`Document`) | Necessário para CRUD, versionamento, auditoria — não cabe em metadados soltos do vetor, que não são consultáveis/editáveis fora do pipeline de indexação. |
| Decisão "preciso buscar? qual tool chamar?" | **Lógica do agente** (orchestrator + function calling nativo do LLM) | Comportamento de tempo de execução, não conhecimento persistente — não deve ser armazenado, apenas executado. |
| Persona, regras de citação, restrições de alucinação (grounding, negative constraints) | **System Prompt** | Comportamento constante do agente, independente de documento ou curso. Pequenas variações por curso (ex.: nome do curso ativo) podem ser injetadas como uma linha dinâmica no prompt, mas a espinha dorsal permanece fixa em código. |
| Curso selecionado pelo usuário / contexto da sessão atual | **Payload da requisição** (`course_id` no `ChatRequest`) | É estado de interação, não conhecimento de domínio. O backend continua stateless (histórico enviado pelo cliente, como hoje) — não introduzir memória de sessão persistente sem uma necessidade real comprovada. |
| Parâmetros de retrieval (top_k, pesos RRF, modelos) | **Configuração** (`config.py` / variáveis de ambiente) | Já é o padrão do projeto; apenas estendido com `DATABASE_URL`, `JWT_SECRET` e afins. |
| Log de interações (pergunta, fontes, tokens) | **Arquivo JSONL** (mantido) | Já funciona, baixo custo, não é consultado por usuário final. Migração para tabela é Fase 8 (opcional), só se o painel admin precisar de agregações que arquivo não resolve bem. |
| Histórico de conversa dentro de uma sessão ativa | **Não persistido no backend** (client-supplied, como na v1) | Evita memória de longo prazo não solicitada; simples e já funciona. |

---

## 6. Estratégia de RAG

### 6.1 Um índice vetorial ou vários? — Decisão: **um único ChromaDB, particionado por metadados**

Avaliei três alternativas:

| Alternativa | Prós | Contras |
|---|---|---|
| **A. Collection única + metadados ricos** (`kb_slug`, `course_id`, `status`) — recomendada | Reaproveita 100% de `vector_store.py`/`hybrid_search.py`; um único índice BM25 (após corrigir D1); busca cross-documento naturalmente cobre perguntas que cruzam Estatuto+Estágio+PPC, que é exatamente o problema relatado | Filtros de metadado precisam ser bem desenhados (mas ChromaDB já suporta `$and`/`$or`/`$in` — usado hoje só para `status`) |
| B. Uma collection por base de conhecimento | Isolamento total, tuning independente por tipo de documento | N conexões, N índices BM25 em memória, buscas que precisam de mais de uma base exigem fan-out + merge manual — mais peças móveis para um corpus que hoje tem ~1.200 chunks, não justificável em escala atual |
| C. Uma collection por curso | Resolve escopo por curso "de graça" | Duplica documentos institucionais (Estatuto, Regimento) em cada curso; documentos multi-curso (PPC comparado entre cursos, se um dia existir) ficam sem lugar natural; não escala bem para "N cursos futuros" |

**Recomendação: A.** O ganho de isolamento das alternativas B/C não compensa a complexidade operacional extra para o volume de dados real do projeto, e a alternativa A já resolve o problema central (respostas que dependem de informação espalhada entre documentos) ao manter tudo em um único pool de busca. Reservo a decisão de criar uma segunda collection apenas para um cenário concreto que **não existe hoje**: um tipo de conteúdo que exija um modelo de embedding diferente (ex.: tabelas puras). Se isso surgir, decide-se então — não antecipar.

Importante: isso significa que a solução para "faltam informações espalhadas entre documentos" **não é fragmentar a recuperação por categoria**, e sim o oposto — manter um funil único e mais completo, e só usar `kb_slug`/`course_id` como filtros *opcionais* de precisão quando fizer sentido (ex.: perguntas explicitamente sobre PPC de um curso), nunca como partição obrigatória.

### 6.2 Chunking

- **Documentos normativos/legais (Estatuto, Regimento, Resoluções, Estágio, PPC como resolução)**: mantém `legal_chunker.py` sem alterações — já é a estratégia correta e testada.
- **Documentos não estruturados por artigo** (ex.: um eventual "Manual do Aluno" em formato de FAQ/tópicos, não de lei): hoje `split_into_articles()` já degrada graciosamente (retorna o documento inteiro como um único chunk se não encontrar nenhum "Art."), o que é ruim para documentos longos — perderia granularidade de recuperação. Proponho um módulo novo, pequeno, `heading_chunker.py`, que segmenta por cabeçalhos Markdown (`#`, `##`) quando o documento não casa com o padrão de artigos, reaproveitando o mesmo contrato de saída (`LegalChunk`/`ChunkMetadata`) para não exigir nenhuma mudança a jusante (indexação, busca, geração continuam agnósticas de qual chunker gerou o chunk).
- Critério de escolha do chunker: no `IngestionService` (Seção 7.2), tenta `legal_chunker` primeiro; se `split_into_articles()` retornar um único bloco para um documento com mais de N tokens (heurística: sem estrutura de artigo detectada), cai para `heading_chunker`. Decisão automática, sem exigir que o admin escolha manualmente.

### 6.3 Metadados

Extensão mínima e retrocompatível de `ChunkMetadata` ([src/chunking/legal_chunker.py](src/chunking/legal_chunker.py)):

```python
@dataclass
class ChunkMetadata:
    hierarchy: list[str] = field(default_factory=list)
    source: str = ""
    category: str = ""
    status: str = "vigente"
    article_id: str = ""
    chunk_index: int = 0
    is_child_chunk: bool = False
    parent_article: str = ""
    kb_slug: str = "regulamentos"   # NOVO — default preserva o comportamento atual
    course_id: str | None = None    # NOVO — None = institucional (aplica a todos os cursos)
```

Chunks já indexados na v1 recebem `kb_slug="regulamentos"` e `course_id=None` por padrão — nenhuma reindexação obrigatória para manter o sistema funcionando.

### 6.4 Filtros e retrieval

O `where_filter` de `hybrid_search.py`/`vector_store.py` evolui de `{"status": "vigente"}` para uma composição:

```python
where_filter = {
    "$and": [
        {"status": "vigente"},  # já existe
        {"$or": [{"course_id": course_id}, {"course_id": None}]},  # novo, só se course_id foi informado
    ]
}
```

`kb_slug` **não** entra como filtro obrigatório por padrão — apenas quando uma Tool específica quiser restringir (ex.: uma futura consulta "me mostra só o PPC" poderia passar `kb_slug="ppc"` explicitamente). Mantém a filosofia da Seção 6.1: pool único, filtro é exceção, não regra.

### 6.5 Reranking

Mantido (`BAAI/bge-reranker-v2-m3`). Correção pontual: `chat/service.py` (ou o futuro `RagTool`) passa a chamar `rerank(..., score_threshold=RERANK_SCORE_THRESHOLD)`, com o novo parâmetro exposto em `config.py` (default sugerido: 0.4, alinhado ao valor citado no TCC1). Isso fecha o gap D3 do diagnóstico sem qualquer mudança de algoritmo.

### 6.6 Estratégia de citações

Mantida como está — cada chunk carrega fonte, categoria, artigo e hierarquia, e o prompt de resposta já exige citação inline. Única extensão: quando a resposta combina RAG + Tool estruturada (ex.: pergunta sobre "quem leciona a disciplina X, segundo o PPC"), o orchestrator consolida `sources` de ambas as origens no mesmo array de resposta, distinguíveis por um campo `origin: "rag" | "professor" | "calendar"`.

### 6.7 Atualização de documentos e versionamento

- `Document.version` incrementa a cada reindexação.
- `Document.superseded_by_document_id` (auto-relacionamento) formaliza "esta resolução foi revogada por aquela" além da heurística textual atual (`revocation_filter.py`, mantida como sinal automático inicial, mas agora confirmável/editável pelo admin).
- Reindexar um documento **apaga** os `chunk_id`s antigos daquele documento no ChromaDB antes de inserir os novos — usando uma tabela `DocumentChunk` como índice desses IDs (ver Seção 7.1). Isso resolve D2.

### 6.8 Prevenção de alucinações

Mantidos: grounding estrito no prompt, negative constraint ("não invente"), filtro hard de vigência, agora reforçado pelo `score_threshold` (D3) e estendido ao domínio estruturado: `ProfessorTool`/`CalendarTool` retornam explicitamente "não encontrado" quando a query não bate com nenhum registro — mesma disciplina de honestidade que já existe no caminho RAG hoje ("Não encontrei informações relevantes...").

### 6.9 Recuperação contextual (multi-turno)

Mantido o padrão atual (histórico enviado pelo cliente, injetado nas mensagens de decisão e de geração). Adição: quando `course_id` é enviado uma vez na conversa, o cliente deve reenviá-lo a cada requisição (o backend continua stateless) — não é necessário criar sessão de servidor para isso.

---

## 7. Infraestrutura Administrativa do Backend

### 7.1 Modelagem de dados

Novo módulo `src/db/` com SQLAlchemy + Alembic. Entidades:

```
Course
  id (PK), code (ex: "ENGCOMP"), name, active, created_at

KnowledgeBase
  id (PK), slug (ex: "regulamentos", "manual_aluno", "ppc"),
  name, description, chunking_strategy ("legal" | "heading"),
  scope ("institutional" | "course_specific")

Document
  id (PK), knowledge_base_id (FK → KnowledgeBase),
  course_id (FK → Course, nullable = institucional),
  title, filename, storage_path, checksum,
  status ("processing" | "indexed" | "failed" | "archived"),
  version (int), revoked (bool), revoked_reason,
  superseded_by_document_id (FK → Document, nullable),
  uploaded_by (FK → AdminUser), uploaded_at, indexed_at

DocumentChunk        # espelho leve dos vetores — não duplica o conteúdo
  id (PK), document_id (FK → Document), chroma_id,
  article_id, hierarchy, token_count

IngestionJob
  id (PK), document_id (FK → Document), stage ("etl"|"chunking"|"embedding"|"indexing"),
  status ("queued"|"running"|"done"|"failed"), error_message,
  started_at, finished_at

Professor
  id (PK), name, email, department, bio, created_at, updated_at

Discipline
  id (PK), course_id (FK → Course), name, code, period, workload

ProfessorDiscipline   # M2M com atributos
  professor_id (FK), discipline_id (FK), semester_year, schedule_text, room

AdminUser
  id (PK), email, password_hash, role ("admin"|"editor"), created_at, last_login
```

Justificativa pontual de cada tabela:

- `KnowledgeBase` como entidade (não enum fixo) porque o painel administrativo pedido na Seção 7 precisa listar/gerenciar categorias — é a materialização em dado do que hoje são strings soltas (`category`) espalhadas em `pdf_converter.py`.
- `DocumentChunk` não guarda o texto do chunk (isso já vive no ChromaDB) — guarda só o `chroma_id`, para permitir apagar vetores órfãos no reindex (resolve D2) e para o admin poder listar "quantos chunks esse documento gerou" sem consultar o ChromaDB diretamente.
- `ProfessorDiscipline` como M2M com atributos (não FK direta em `Professor`) porque um professor pode lecionar mais de uma disciplina e uma disciplina pode ter mais de um professor ao longo dos semestres — modelagem correta desde o início evita migração dolorosa depois.
- `AdminUser` é deliberadamente separado de qualquer futuro sistema de conta de aluno — o TCC1/EVOLUTION.md (decisão D6) já decidiu que autenticação de usuário final está fora de escopo; isto não contradiz aquela decisão, apenas resolve o problema **novo** (auth para quem administra conteúdo), que antes não existia porque não havia painel admin.

### 7.2 Serviços

| Serviço | Responsabilidade | Reaproveita |
|---|---|---|
| `IngestionService` (`src/ingestion/`) | `process_document(path, course_id?, kb_id) -> Document` — orquestra ETL → chunking (legal ou heading) → embedding → indexação → grava `DocumentChunk`s → atualiza `Document.status` | 100% de `pdf_converter.py`, `legal_chunker.py`, `vector_store.py` — só adiciona a orquestração e a gravação em banco que antes eram feitas via `print`/log no script |
| `DocumentService` (`src/documents/`) | CRUD de `Document`, versionamento, trigger de reindex, exclusão (remove chunks via `DocumentChunk`) | Reaproveita `resolve_pdf`/download atual, troca só a fonte de listagem (filesystem → DB) |
| `CourseService` (`src/courses/`) | CRUD de `Course` | novo, simples |
| `KnowledgeBaseService` (`src/knowledge_bases/`) | CRUD de `KnowledgeBase` (dado que muda raramente, poucos endpoints) | novo, simples |
| `ProfessorService` (`src/professors/`) | CRUD de `Professor`/`Discipline`/`ProfessorDiscipline` | novo — implementa finalmente a decisão D9 do `EVOLUTION.md` |
| `AdminAuthService` (`src/admin/`) | Login de `AdminUser`, emissão/validação de JWT de curta duração | Segue o mesmo padrão de `Security`/`Depends` já usado em [src/auth.py](src/auth.py) |

### 7.3 APIs

```
# Documentos (admin, protegido por AdminUser + role)
POST   /admin/documents                 upload multipart (course_id?, knowledge_base_id, title)
GET    /admin/documents                 lista/filtra por curso, kb, status
PATCH  /admin/documents/{id}            atualiza metadados, marca revogado, define superseded_by
DELETE /admin/documents/{id}            remove documento + chunks associados
POST   /admin/documents/{id}/reindex    força novo ETL→chunk→embed, substituindo chunks antigos

# Cursos
POST   /admin/courses | GET/PATCH/DELETE /admin/courses/{id}   (admin)
GET    /courses                                                 (público — popula seletor de curso)

# Bases de conhecimento
POST   /admin/knowledge-bases | GET/PATCH/DELETE .../{id}       (admin)

# Professores
POST   /admin/professors | GET/PATCH/DELETE /admin/professors/{id}   (admin)
POST   /admin/professors/{id}/disciplines                              (admin — associação)
GET    /professors?course_id=&discipline=&name=                        (público)

# Ingestão
GET    /admin/ingestion-jobs?document_id=                        (admin — visibilidade do pipeline)

# Auth admin
POST   /admin/auth/login                                          (email+senha → JWT)
```

Todas as rotas `/admin/*` usam uma dependência nova (`get_current_admin`, análoga a `get_api_key`) que valida o JWT e o papel do usuário — a rota pública `/chat`, `/documents`, `/logs` continua exatamente como está, protegida pela `x-api-key` (decisão D6 do `EVOLUTION.md` preservada).

### 7.4 Fluxo completo de upload de um documento

1. Admin autenticado faz `POST /admin/documents` (multipart: arquivo + `course_id?` + `knowledge_base_id` + `title`).
2. Backend salva o arquivo em `data/raw/{document_id}/{filename}` (mantém a filosofia de armazenamento local já usada pelo projeto — sem introduzir S3/object storage, que seria overengineering para este estágio), cria `Document(status="processing")` e `IngestionJob(stage="etl", status="running")`.
3. `IngestionService.process_document()`:
   a. **ETL** — `pdf_converter.convert_pdf_to_markdown()` (mantido, `pymupdf4llm`).
   b. **Revogação** — `revocation_filter.analyze_revocation()` (mantido).
   c. **Chunking** — `legal_chunker.chunk_document()`, com fallback para `heading_chunker` se não detectar artigos (Seção 6.2).
   d. **Embedding + indexação** — `vector_store.index_chunks()`, agora recebendo `kb_slug`/`course_id` para popular os novos campos de metadado.
   e. Grava um `DocumentChunk` por chunk (`chroma_id`, `article_id`, `token_count`) para rastreabilidade e para permitir exclusão seletiva depois.
   f. Se um documento antigo com o mesmo `document_id` (reindex) tinha `DocumentChunk`s de uma versão anterior, **apaga esses `chroma_id`s do ChromaDB antes de inserir os novos** (fecha D2).
   g. Atualiza `Document.status="indexed"`, `Document.indexed_at`, `IngestionJob.status="done"`.
4. Em caso de falha em qualquer etapa: `IngestionJob.status="failed"` + `error_message`, `Document.status="failed"` — visível via `GET /admin/ingestion-jobs`, e o admin pode disparar `POST /admin/documents/{id}/reindex` para tentar de novo.
5. **Disponibilização para consulta**: a busca densa já enxerga o documento imediatamente (consulta o ChromaDB ao vivo). A busca esparsa (BM25) só o enxergaria depois de um restart do processo — por isso o `IngestionService`, ao concluir com sucesso, chama `get_search_engine().reload()` (novo método, recarrega `self.chunks` e reconstrói o índice BM25 em memória). Isso fecha D1 e é o elo que faltava para o fluxo "upload → disponível para consulta" funcionar de ponta a ponta sem reiniciar o servidor.

Para o volume de documentos deste projeto (dezenas, não milhares), processar a ingestão de forma **síncrona** dentro do próprio request (com timeout generoso) é suficiente — uma fila assíncrona (Celery/RQ) é adiada deliberadamente para não introduzir infraestrutura sem necessidade comprovada (ver Seção 9).

---

## 8. Evolução do Agente

| Dimensão | Como evolui |
|---|---|
| **Qualidade das respostas** | Melhora principalmente por **cobertura**: mais documentos (Manual do Aluno) e mais tipos de dado (professores, calendário) no mesmo funil de busca. O algoritmo de recuperação em si já é adequado (Seção 2.1). |
| **Recuperação de contexto** | Correção de D1 (BM25 desatualizado) e D3 (threshold não aplicado) já elevam a qualidade sem nenhuma mudança de algoritmo — são bugs, não limitações de design. |
| **Precisão** | `score_threshold` no rerank evita injetar contexto de baixa relevância; filtro `course_id` opcional reduz ruído quando o usuário informa o curso. |
| **Uso de Tools** | Sai de "1 decisão binária hardcoded" para "N tools registradas, escolhidas via function calling nativo do modelo" — `RagTool` (normas) e `ProfessorTool` (estruturado) no MVP da v2; `CalendarTool` como extensão natural do mesmo padrão. |
| **Tomada de decisão do agente** | Function calling nativo da OpenAI substitui o parsing manual de JSON (D5) — mais robusto (não depende de regex para tirar ```` ``` ````), suporta múltiplas tools e chamadas paralelas nativamente. |
| **Combinação entre documentos e dados estruturados** | O orchestrator pode chamar `RagTool` e `ProfessorTool` na mesma resposta (ex.: "quem leciona Cálculo I e onde isso está previsto no PPC?") e consolidar as duas fontes num único `sources[]` com campo `origin`. |
| **Redução de alucinações** | Mantém grounding + negative constraints do prompt atual; estende a mesma disciplina de "diga que não sabe" para as tools estruturadas. |
| **Referências aos documentos** | Mantido o padrão atual de citação inline com fonte/artigo/hierarquia; estendido com `origin` para diferenciar fonte RAG de fonte estruturada na resposta. |

---

## 9. Roadmap de Implementação

| Fase | Objetivo | Principais entregas | Impacto arquitetural | Dependências | Prioridade | Complexidade |
|---|---|---|---|---|---|---|
| **0 — Fundação de dados** | Introduzir a camada relacional | PostgreSQL + SQLAlchemy + Alembic; modelos `Course`, `KnowledgeBase`, `Document`, `DocumentChunk`, `IngestionJob`, `Professor`, `Discipline`, `ProfessorDiscipline`, `AdminUser`; seed de 1 curso (Eng. Computação) + 1 KB (regulamentos) | Alto — nova peça de infra (novo serviço no `docker-compose.yml`) | Nenhuma | **Alta** | Média |
| **1 — Refatoração da ingestão** | Extrair lógica reaproveitável dos scripts CLI | `src/ingestion/service.py`; correção D1 (reload BM25) e D2 (limpeza de chunks órfãos); extensão de `ChunkMetadata`/`flat_meta` com `kb_slug`/`course_id` | Médio — scripts CLI viram wrappers finos do serviço | Fase 0 | **Alta** | Média |
| **2 — API administrativa de documentos** | Permitir upload/gestão sem tocar em CLI | `POST/GET/PATCH/DELETE /admin/documents`, `.../reindex`; `AdminAuthService` (JWT) | Médio-alto — primeira rota autenticada por papel | Fases 0, 1 | **Alta** | Média-Alta |
| **3 — Corpo docente** | Fechar a decisão D9 do `EVOLUTION.md` | `ProfessorService`, `Discipline`, endpoints admin + `GET /professors` público | Baixo — entidade nova, sem tocar em RAG | Fase 0 | **Média** | Baixa-Média |
| **4 — Orquestração multi-tool** | Resolver D5 (decisão binária → tools) | `src/agent/orchestrator.py`, `RagTool`, `ProfessorTool`; elimina duplicação `run_chat`/`stream_chat` (D4); aplica `score_threshold` (D3) | Alto — é o núcleo da qualidade de resposta | Fase 3 (para ter uma 2ª tool de fato) | **Alta** | Média-Alta |
| **5 — Expansão de conteúdo** | Resolver a causa raiz do problema relatado | Ingestão de Manual do Aluno (quando disponível) via a nova API admin; `heading_chunker.py` para documentos não-legais | Baixo — reaproveita 100% do pipeline | Fases 1, 2 | **Alta** | Baixa |
| **6 — Escopo por curso** | Habilitar seleção de curso end-to-end | `course_id` opcional no `ChatRequest`; filtro `$or [course_id, null]`; `GET /courses` | Baixo — modelo de dados já suporta desde a Fase 0 | Fases 0, 4 | **Média** | Baixa |
| **7 — Calendário acadêmico** (condicional) | Só se houver demanda real validada nos logs de uso | Tabela `AcademicEvent` + `CalendarTool` | Baixo | Fase 4 | **Baixa-Média** | Baixa |
| **8 — Observabilidade admin** (opcional) | Analytics agregadas por curso/documento | Avaliar migração de `queries.jsonl` para tabela `QueryLog` | Baixo | Fase 0 | **Baixa** | Baixa |

---

## 10. Plano Técnico de Implementação

Ordem recomendada, com granularidade de execução:

### 10.1 `db-schema` — Introduzir PostgreSQL + SQLAlchemy + Alembic
- **Problema resolvido**: D6 (zero banco relacional), pré-requisito de tudo mais.
- **Justificativa técnica**: CRUD de documentos/cursos/professores com relações (FK, M2M) e integridade referencial não é modelável de forma sã em arquivos ou em metadados de vetor.
- **Componentes**: novo `src/db/` (engine, session, `Base`), `alembic/`, novo serviço `postgres` em `docker-compose.yml` (mesmo padrão já usado para `chromadb`).
- **Impacto arquitetural**: Alto (nova infraestrutura), mas isolado — nenhum módulo existente muda de comportamento nesta etapa.
- **Esforço**: 1–2 dias.
- **Riscos**: baixo, é aditivo. Cuidado com credenciais em `.env`/`docker-compose.yml` (seguir o padrão já usado para `TUNNEL_TOKEN`).

### 10.2 `ingestion-service` — Extrair `IngestionService` dos scripts CLI
- **Problema resolvido**: D7 (ingestão só via CLI, sobre o corpus inteiro).
- **Justificativa**: pré-requisito da API admin; sem essa extração, a API teria que duplicar a lógica dos scripts.
- **Componentes**: `src/ingestion/service.py`, `scripts/run_etl.py`/`run_indexing.py` refatorados para chamar o serviço.
- **Impacto**: Médio — os scripts continuam funcionando (retrocompatível), só mudam por dentro.
- **Esforço**: 1 dia.
- **Riscos**: baixo — cobrir com um teste manual rodando `run_etl.py`+`run_indexing.py` de ponta a ponta antes/depois para garantir paridade de saída.

### 10.3 `bm25-reload` + `chunk-orphan-cleanup` — Corrigir D1 e D2
- **Problema resolvido**: staleness do índice esparso e vetores órfãos em reindexação.
- **Justificativa**: bloqueia diretamente o fluxo "upload → disponível para consulta" pedido na Seção 7.
- **Componentes**: `HybridSearchEngine.reload()` em [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py); uso da tabela `DocumentChunk` no `IngestionService` para apagar `chroma_id`s antigos antes de reindexar.
- **Impacto**: Baixo — mudança localizada, sem quebrar contrato de API.
- **Esforço**: meio dia.
- **Riscos**: baixo.

### 10.4 `rerank-threshold` — Corrigir D3
- **Problema resolvido**: mecanismo de corte de relevância descrito na metodologia não estava ativo.
- **Componentes**: novo `RERANK_SCORE_THRESHOLD` em `config.py`; passagem do parâmetro no ponto de chamada de `rerank()`.
- **Impacto**: Muito baixo.
- **Esforço**: 1 hora.
- **Riscos**: pode reduzir recall se o threshold for calibrado errado — validar com o `golden_dataset.json` existente (`scripts/run_eval.py`) antes/depois.

### 10.5 `admin-documents-api` — Endpoints administrativos de documentos
- **Problema resolvido**: D7, viabiliza o painel admin futuro.
- **Componentes**: `src/documents/` ganha camada de serviço (`DocumentService`) sobre o novo modelo `Document`; novos endpoints `/admin/documents*`; `src/admin/` com `AdminAuthService` (JWT) reaproveitando o padrão `Security`/`Depends` de [src/auth.py](src/auth.py).
- **Impacto**: Médio-alto — primeira rota com autenticação por papel do sistema.
- **Esforço**: 2–3 dias.
- **Riscos**: segurança — revisar CORS (D10) antes de expor rotas de escrita administrativas.

### 10.6 `professors-module` — Fecha a decisão D9
- **Problema resolvido**: pergunta explícita do usuário sobre professores; valida a modelagem estruturada.
- **Componentes**: modelos `Professor`/`Discipline`/`ProfessorDiscipline`; `ProfessorService`; endpoints admin + `GET /professors` público.
- **Impacto**: Baixo — módulo novo e isolado.
- **Esforço**: 1–2 dias.
- **Riscos**: baixo.

### 10.7 `agent-orchestrator` — Resolve D4 e D5 (núcleo da evolução do agente)
- **Problema resolvido**: decisão binária hardcoded (D5) e duplicação `run_chat`/`stream_chat` (D4).
- **Justificativa técnica**: function calling nativo é o mecanismo padrão da indústria para agentes multi-tool; parsing manual de JSON não escala e já demonstrou fragilidade (necessidade de patch ad hoc para ```` ``` ````).
- **Componentes**: `src/agent/orchestrator.py`, `src/agent/tools/rag_tool.py` (wrap de `hybrid_search`+`hyde`+`reranker`, sem alterar essas implementações), `src/agent/tools/professor_tool.py`; `src/chat/service.py` simplificado para delegar ao orchestrator, unificando streaming e não-streaming em um único caminho de execução parametrizado.
- **Impacto**: Alto — é a mudança mais profunda do plano, mas isolada em `src/agent/` + `src/chat/`, sem tocar em retrieval/generation.
- **Esforço**: 3–5 dias.
- **Riscos**: regressão de comportamento (o modelo pode decidir chamar tools de forma diferente do JSON manual atual) — mitigar rodando `scripts/run_eval.py` (RAGAS) antes/depois como regressão automatizada.

### 10.8 `content-expansion` — Ingestão de novos documentos
- **Problema resolvido**: causa raiz do problema relatado pelo usuário (cobertura insuficiente).
- **Componentes**: uso da API admin (10.5) para subir Manual do Aluno (quando obtido); `heading_chunker.py` novo, usado apenas como fallback automático.
- **Impacto**: Baixo — conteúdo, não código de infraestrutura.
- **Esforço**: depende da quantidade de documentos novos a curar.
- **Riscos**: qualidade do documento fonte (se for só imagem escaneada sem OCR, `pymupdf4llm` pode extrair mal — se isso for observado empiricamente, é o único cenário que justificaria avaliar Docling, e só então).

### 10.9 `course-scoping` — Curso opcional ponta a ponta
- **Problema resolvido**: D9 (falta de dimensão de curso).
- **Componentes**: `course_id` opcional em `ChatRequest`; filtro `$or` no retrieval; `GET /courses`.
- **Impacto**: Baixo — aditivo e retrocompatível.
- **Esforço**: 1 dia.
- **Riscos**: baixo.

### 10.10 (opcional, condicional a validação de demanda) `calendar-module`
- Mesma lógica de `professors-module`, aplicada a eventos de calendário. Só entra no roadmap se os logs de `GET /logs/stats` (já existente) mostrarem volume relevante de perguntas sobre prazos/datas não respondidas hoje.

---

## Resumo executivo

O pipeline de recuperação atual (chunking legal, busca híbrida, HyDE, cross-encoder) está bem desenhado e **não deve ser reescrito**. O problema relatado — respostas incompletas por informação espalhada — tem duas causas reais, verificadas nos dados: (1) conteúdo genuinamente ausente do corpus (Manual do Aluno, Calendário, Corpo Docente), e (2) ausência de infraestrutura para adicionar conteúdo e tipos de dado novos sem reprocessar tudo manualmente via CLI. A resposta arquitetural não é fragmentar a base em múltiplos índices vetoriais — é **manter um único funil de busca bem alimentado** para tudo que é texto normativo, e **introduzir uma camada relacional + orquestração multi-tool** para tudo que é dado estruturado (professores, calendário) e para a operação administrativa (upload, versionamento, cursos). A peça genuinamente nova de infraestrutura é o banco de dados relacional; todo o resto é extensão pontual e corrigida do que já existe.
