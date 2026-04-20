# EVOLUTION.md — Registro de Evolução do Projeto

> **Propósito:** Documento vivo para mapear a evolução técnica, decisões de arquitetura e trade-offs do projeto. Serve como base para a escrita final do TCC. Atualize este arquivo ao fim de cada sprint ou decisão relevante.

---

## Contexto Geral

**Título do TCC:** Desenvolvimento de uma Solução de Geração Aumentada por Recuperação (RAG) para Automatizar a Consulta de Documentos Normativos da UNIVASF

**Problema:** Alunos e servidores têm dificuldade em localizar respostas precisas em 48 documentos normativos (estatuto, regimentos, resoluções da PROEN, PROEX e PRPPGI). A consulta manual é lenta, sujeita a erros e desconsidera documentos revogados.

**Solução proposta:** Sistema Advanced RAG com pipeline de recuperação em dois estágios, chunking semântico-hierárquico baseado na estrutura legal, e interface de chat com citação obrigatória das fontes.

---

## Sprint 0 — Implementação do Pipeline RAG Completo

**O que foi feito:**
- Implementados todos os 8 módulos do pipeline em `src/`: ETL, chunking, indexing, retrieval, reranker, generation, evaluation, Streamlit UI.
- Verificação funcional: 48 PDFs detectados, todos os módulos importam sem erros, chunker testado.
- Interface de usuário inicial: Streamlit com exibição de fontes e scores.

**Resultado do teste do chunker:**
- Dado um trecho com 3 artigos legais: 4 chunks gerados (1 preâmbulo + 3 artigos)
- IDs extraídos corretamente: Art. 1º, Art. 2º, Art. 3º
- Hierarquia rastreada: mudança de CAPÍTULO I para CAPÍTULO II detectada
- Parágrafos mantidos com seus artigos: § 1º e § 2º agrupados com Art. 1º

---

## Sprint 1 — Migração para FastAPI + Agente Leve

**O que foi feito:**
- Migração do pipeline RAG para API REST com FastAPI.
- Criação de agente leve com decisão de "precisa buscar?" antes de ativar o pipeline RAG.
- Endpoint `POST /chat/` com suporte a histórico de conversa.
- Streamlit rewired para consumir a API (descoberto como inadequado para deploy — ver decisão abaixo).

**Testes validados:**
- Health check: `GET /health → 200`
- Cumprimento (sem busca): resposta direta sem acionar RAG
- Pergunta sobre normas: pipeline completo acionado, fontes retornadas
- Follow-up com histórico: agente buscou com contexto corretamente

---

## Sprint 2 — Infraestrutura Docker + Deploy

**O que foi feito:**
- `Dockerfile` e `docker-compose.yml` criados para orquestrar API + ChromaDB.
- Script `deploy.sh` para atualização contínua na VPS.
- Configuração do Cloudflare Tunnel para exposição segura da API sem abrir portas.
- ChromaDB migrado de cliente local para cliente HTTP (para comunicação entre containers).

---

## Sprint 3 — Segurança + Frontend React

**O que foi feito:**
- Proteção da API com `x-api-key` header (middleware FastAPI).
- Frontend React construído consumindo a API FastAPI.
- Variável `VITE_API_KEY` no frontend para autenticação.

---

## Decisões de Arquitetura (para o TCC)

### D1 — Chunking por Artigo (Semântico-Hierárquico)

**Decisão:** Cada chunk corresponde a um Artigo completo (caput + parágrafos + incisos). Artigos longos são divididos com herança do caput nos chunks filhos.

**Alternativa rejeitada:** Chunking por tamanho fixo de caracteres/tokens (ex: 512 tokens com overlap).

**Justificativa:** Documentos normativos têm estrutura semântica rígida definida pela Lei Complementar 95/1998. Dividir um artigo ao meio separa a regra (caput) da exceção (parágrafo), o que destrói o contexto jurídico. O fenômeno "Lost in the Middle" é especialmente crítico em textos legais. A segmentação por artigo é a unidade mínima de sentido jurídico.

**Impacto:** Chunks com tamanho variável, mas semanticamente íntegros. Metadados ricos por chunk: hierarquia (Título > Capítulo > Seção), artigo, fonte, status de vigência.

---

### D2 — Busca Híbrida (Dense + BM25 + RRF)

**Decisão:** Combinação de busca densa vetorial (ChromaDB/HNSW) com busca esparsa por palavras-chave (BM25), fundidas via Reciprocal Rank Fusion (RRF).

**Alternativa rejeitada:** Apenas busca vetorial (dense-only).

**Justificativa:** Documentos normativos contêm termos técnicos exatos (ex: "Art. 45", "PSPVO", siglas de programas) que a busca semântica vetorial pode não recuperar bem por não ter visto esses termos no treinamento do embedding. O BM25 garante precisão em termos exatos, enquanto o dense captura a intenção semântica da pergunta. O RRF funde os rankings sem precisar calibrar pesos.

**Parâmetro:** Top-50 candidatos antes do reranker (prioriza recall).

---

### D3 — Cross-Encoder Reranker (Two-Stage Retrieval)

**Decisão:** Após recuperar top-50 candidatos, aplicar modelo cross-encoder (`BAAI/bge-reranker-v2-m3`) para reordenar e selecionar top-5 para o LLM.

**Alternativa rejeitada:** Enviar os top-50 diretamente ao LLM (sem reranking).

**Justificativa:** Bi-encoders (embeddings) avaliam query e documento separadamente — perdem nuances sintáticas finas. Cross-encoders avaliam o par (query, documento) simultaneamente, com atenção cruzada completa, capturando relevância contextual profunda. Reduz drasticamente o contexto enviado ao LLM (top-50 → top-5), reduzindo custo e o risco de "distração" do modelo por contexto irrelevante.

---

### D4 — Filtro de Documentos Revogados

**Decisão:** Pré-filtrar documentos com `status: revogado` antes da busca vetorial. O status é detectado automaticamente por regex no nome do arquivo e no conteúdo.

**Justificativa:** "Alucinação jurídica" — uma norma tecnicamente correta mas revogada é pior do que nenhuma resposta. O sistema precisa garantir segurança jurídica. Detecção automática evita manutenção manual da lista de revogados.

---

### D5 — Streamlit → React (Mudança de Frontend)

**Decisão:** Abandonar o Streamlit como UI principal e adotar React consumindo a API FastAPI.

**Justificativa:** O Streamlit foi útil para prototipagem rápida, mas limitado para deploy separado, para integrações futuras e para uma experiência de usuário profissional. Com a migração para FastAPI, o backend virou uma API REST desacoplada — qualquer frontend pode consumi-la. O React permite um design mais adequado para apresentação do TCC.

---

### D6 — Auth por API Key (em vez de JWT + email @univasf)

**Decisão:** Proteção da API via header `x-api-key` estático, sem sistema de registro/login de usuários.

**Alternativa planejada:** Auth completo com JWT, registro com validação de email `@univasf.edu.br`, perfis de usuário (aluno/professor).

**Justificativa:** O auth completo é out of scope para o MVP do TCC. O foco acadêmico é o pipeline RAG em si (chunking, retrieval, reranking, avaliação). A API key é suficiente para controlar acesso em ambiente de demonstração/avaliação. O sistema de usuários pode ser adicionado em versões futuras.

---

### D7 — Avaliação com RAGAS (LLM-as-a-Judge)

**Decisão:** Avaliação automatizada com framework RAGAS usando 4 métricas: Faithfulness, Answer Relevance, Context Precision, Context Recall.

**Golden Dataset:** 15 perguntas baseadas em dúvidas reais de alunos (matrícula, trancamento, estágio, colação de grau, dispensa de componentes).

**Justificativa:** Avaliação manual por humanos é subjetiva, cara e não reproduzível. O paradigma LLM-as-a-Judge com métricas estruturadas permite comparação objetiva entre configurações do pipeline (ex: com e sem reranker, diferentes valores de top-k). Faithfulness é a métrica mais crítica para o domínio jurídico — detecta alucinação.

---

### D9 — Módulo de Professores: Curadoria Manual (em vez de upload público)

**Decisão:** Cadastro de professores e cronogramas via endpoint protegido por API key (curadoria pelo próprio desenvolvedor), com leitura pública via `GET /professors`. Dados entram como JSON estruturado direto no ChromaDB em uma coleção separada (`type: schedule`), sem pipeline ETL de PDF.

**Alternativas rejeitadas:**

- *Upload público de PDF:* qualquer pessoa enviaria PDFs que seriam processados e indexados automaticamente. Risco alto de poluição do ChromaDB com dados incorretos — numa demonstração na banca, o agente poderia citar informação errada com confiança, o que é pior que não ter o dado.
- *Submissão pública + moderação:* mais robusto, mas adiciona complexidade de fila de aprovação sem auth — over-engineering para o escopo do TCC.

**Justificativa:** A contribuição técnica central do TCC já está no pipeline de normas. O módulo de professores serve para demonstrar que o agente opera sobre múltiplas coleções (normas + cronogramas) e escolhe a ferramenta certa conforme a intenção da pergunta. Para essa demonstração, 5–10 professores cadastrados manualmente são suficientes. Dado estruturado (campos explícitos: nome, disciplina, horários, datas de prova) também produz retrieval mais preciso que texto livre extraído de PDF.

**Como fica a arquitetura:**
- `POST /professors` (protegido) — cadastra/atualiza professor, indexa no ChromaDB com `type: schedule`
- `GET /professors` (público) — lista professores cadastrados
- Agente ganha `ScheduleTool` que busca na coleção de cronogramas
- `LegalTool` permanece inalterada para normas

**Why:** Sem auth no sistema, a única forma segura de garantir qualidade dos dados é controle manual. Para a fase de validação do TCC isso é suficiente e elimina risco de demonstração com dados ruins.

**How to apply:** Ao implementar, manter os dois endpoints separados no roteamento e usar metadado `type` no ChromaDB para isolar as buscas por coleção.

---

### D8 — CloudFlare Tunnel (em vez de expor porta direta)

**Decisão:** Usar Cloudflare Tunnel para expor a API sem abrir portas públicas na VPS.

**Justificativa:** Sem necessidade de configurar firewall, SSL/TLS gerenciado pelo Cloudflare, IP da VPS não fica exposto. Ideal para demonstração durante a banca.

---

## Estado Atual do MVP

| Componente | Status | Notas |
|------------|--------|-------|
| ETL (PDF → Markdown → Chunks) | ✅ Pronto | 48 PDFs, chunking por artigo |
| Indexing (ChromaDB + embeddings) | ✅ Pronto | `text-embedding-3-large` |
| Retrieval híbrido + RRF | ✅ Pronto | Top-50 candidatos |
| Reranker cross-encoder | ✅ Pronto | `BAAI/bge-reranker-v2-m3`, top-5 |
| Geração com citações (GPT-4o) | ✅ Pronto | Temperatura 0.1 |
| FastAPI + endpoint /chat/ | ✅ Pronto | Com histórico de conversa |
| Proteção por API key | ✅ Pronto | Header `x-api-key` |
| Docker + deploy VPS | ✅ Pronto | docker-compose + deploy.sh |
| Cloudflare Tunnel | ✅ Pronto | URL pública estável |
| Frontend React | ✅ Pronto | Consome API FastAPI |
| Avaliação RAGAS | ✅ Pronto | 15 perguntas golden dataset |
| Auth JWT + email @univasf | ❌ Fora do escopo MVP | Decisão D6 |
| Módulo de professores (curadoria) | 🔜 Planejado | Decisão D9 |

---

## Itens Pendentes / Próximos Passos

> Atualize esta seção conforme o projeto avança.

- [ ] Implementar módulo de professores (Decisão D9): `POST /professors`, `GET /professors`, `ScheduleTool` no agente
- [ ] Rodar avaliação RAGAS completa e registrar métricas finais aqui
- [ ] Comparar baseline (sem reranker) vs sistema completo nas métricas RAGAS
- [ ] Documentar exemplos reais de perguntas e respostas para o TCC
- [ ] Registrar custo de tokens médio por consulta (disponível via `GET /logs/stats`)

---

## Métricas Alvo (RAGAS)

| Métrica | Target | Resultado Final |
|---------|--------|-----------------|
| Faithfulness | > 0.8 | — |
| Answer Relevance | > 0.8 | — |
| Context Precision | > 0.7 | — |
| Context Recall | > 0.7 | — |

> Preencher após rodar `python scripts/run_eval.py`
