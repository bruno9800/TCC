# RAG UNIVASF — Assistente Normativo

Sistema **Advanced RAG** para consulta inteligente dos documentos normativos da UNIVASF (Universidade Federal do Vale do São Francisco).

> **TCC** — Desenvolvimento de uma Solução de Geração Aumentada por Recuperação (RAG) para Automatizar a Consulta de Documentos Normativos da UNIVASF

---

## Arquitetura

```
┌─────────────┐    ┌───────────────┐    ┌─────────────┐
│  48 PDFs    │───▶│  ETL Pipeline  │───▶│  Chunks     │
│  (Normas)   │    │  PDF → MD      │    │  JSONL      │
└─────────────┘    └───────────────┘    └──────┬──────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │  ChromaDB       │
                                      │  + Embeddings   │
                                      └────────┬───────┘
                                               │
                    ┌──────────────────────────┤
                    ▼                          ▼
            ┌──────────────┐          ┌──────────────┐
            │  BM25         │          │  Dense Search │
            │  (Esparsa)    │          │  (Vetorial)   │
            └──────┬───────┘          └──────┬───────┘
                   │       Fusion (RRF)       │
                   └───────────┬─────────────┘
                               ▼
                      ┌────────────────┐
                      │  Reranker       │
                      │  Cross-Encoder  │
                      │  Top-50 → Top-5 │
                      └────────┬───────┘
                               ▼
                      ┌────────────────┐
                      │  LLM (GPT-4o)  │
                      │  + Citações     │
                      └────────┬───────┘
                               ▼
                      ┌────────────────┐
                      │  FastAPI        │
                      │  POST /chat/    │
                      └────────┬───────┘
                               ▼
                      ┌────────────────┐
                      │  React Frontend │
                      └────────────────┘
```

---

## Estrutura dos Módulos

### `src/etl/` — Extração e Transformação

| Arquivo | Descrição |
|---------|-----------|
| `pdf_converter.py` | Converte PDFs para Markdown (pymupdf4llm), preservando cabeçalhos, tabelas e hierarquia visual. Classifica cada documento por categoria (Estatuto, Regimento, Resolução PROEN/PROEX/PRPPGI). |
| `revocation_filter.py` | Detecta documentos revogados via regex no nome do arquivo e no conteúdo. Marca cada documento com `status: vigente` ou `status: revogado`. |

### `src/chunking/` — Segmentação Semântica

| Arquivo | Descrição |
|---------|-----------|
| `legal_chunker.py` | Chunking semântico-hierárquico baseado na estrutura legal (Lei Complementar 95/1998). Cada chunk = um Artigo completo (caput + parágrafos + incisos). Artigos longos são divididos com herança de contexto do caput. Metadados por chunk: hierarquia, fonte, categoria, status de vigência. |

### `src/indexing/` — Indexação Vetorial

| Arquivo | Descrição |
|---------|-----------|
| `vector_store.py` | Gera embeddings via OpenAI (`text-embedding-3-large`) e armazena no ChromaDB com persistência em disco. Suporta filtros por metadados. |

### `src/retrieval/` — Recuperação em Dois Estágios

| Arquivo | Descrição |
|---------|-----------|
| `hybrid_search.py` | Combina busca densa (vetorial) com busca esparsa (BM25) via Reciprocal Rank Fusion (RRF). Pré-filtra documentos revogados. Retorna top-50 candidatos. |
| `reranker.py` | Cross-Encoder (`BAAI/bge-reranker-v2-m3`) reordena os 50 candidatos por relevância contextual, selecionando top-5 para o LLM. |

### `src/generation/` — Geração de Respostas

| Arquivo | Descrição |
|---------|-----------|
| `generator.py` | Prompt com persona de assistente jurídico universitário, injeta os 5 chunks recuperados como contexto, gera respostas com citações obrigatórias (Artigo + Norma). GPT-4o com temperatura 0.1. |

### `src/chat/` — API de Chat

| Arquivo | Descrição |
|---------|-----------|
| `router.py` | Endpoint `POST /chat/` com suporte a histórico de conversa. |
| `service.py` | Agente leve — decide quando acionar o pipeline RAG ou responder diretamente. |
| `schemas.py` | Pydantic models (`ChatRequest`, `ChatResponse`, `SourceInfo`). |

### `src/evaluation/` — Avaliação de Qualidade

| Arquivo | Descrição |
|---------|-----------|
| `golden_dataset.json` | 15 perguntas/respostas baseadas em dúvidas reais de alunos. |
| `ragas_eval.py` | Avaliação com RAGAS: Faithfulness, Answer Relevance, Context Precision, Context Recall. |

### `src/main.py` — Entrypoint FastAPI

### `src/config.py` — Configurações Centralizadas

---

## Documentos Fonte

48 PDFs organizados em `regimentos_estatutos_resolucoes/`:

| Diretório | Qtd | Conteúdo |
|-----------|-----|----------|
| raiz | 2 | Estatuto e Regimento Geral da UNIVASF |
| `PROEN/` | 37 | Resoluções da Pró-Reitoria de Ensino |
| `PROEX/` | 7 | Documentos da Pró-Reitoria de Extensão |
| `PRPPGI/` | 4 | Resoluções da Pró-Reitoria de Pesquisa e Pós-Graduação |

---

## Segurança

A API exige o header `x-api-key` em todas as requisições ao `/chat/`. Configure a variável `TCC_API_KEY` no `.env`.

Ver [API.md](./API.md) para referência completa dos endpoints.

---

## Deploy

Ver [DEPLOY.md](./DEPLOY.md) para instruções de deploy em VPS (Docker) ou localmente.

---

## Avaliação (RAGAS)

```bash
python scripts/run_eval.py
```

Executa o pipeline para as 15 perguntas do golden dataset e calcula as métricas. Relatório salvo em `data/ragas_report_<timestamp>.json`.

| Métrica | O que mede | Target |
|---------|-----------|--------|
| Faithfulness | Resposta deriva apenas dos documentos? (detecta alucinação) | > 0.8 |
| Answer Relevance | Resposta atende à dúvida do usuário? | > 0.8 |
| Context Precision | Documentos relevantes apareceram no topo do ranking? | > 0.7 |
| Context Recall | O sistema encontrou toda a informação necessária? | > 0.7 |
