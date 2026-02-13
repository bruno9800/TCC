Com base na análise do TCC "Desenvolvimento de uma Solução de Geração Aumentada por Recuperação (RAG) para Automatizar a Consulta de documentos normativos da UNIVASF", elaborei este guia de execução passo a passo.


ESSE É O MEU TCC EM FORMATO MD

---

# Guia de Implementação: Sistema RAG para Documentos Normativos e Jurídicos

Este guia descreve a arquitetura e o fluxo de execução para a construção de um assistente virtual baseado em **Advanced RAG**, especializado em documentos hierárquicos (leis, estatutos, regimentos). O diferencial desta arquitetura é a preservação da integridade semântica dos dispositivos legais através de estratégias de *chunking* estruturado.

## 1. Visão Geral da Arquitetura

O sistema segue o paradigma **Advanced RAG** com um pipeline de recuperação em dois estágios (Two-Stage Retrieval) e validação via **LLM-as-a-Judge**.

* **Entrada:** PDFs de normas e regulamentos.
* 
**Processamento:** Conversão para Markdown estruturado e segmentação hierárquica baseada em leis de estruturação legislativa (ex: Lei Complementar n.º 95/1998 ).


* **Recuperação:** Busca vetorial densa seguida de reordenação (Reranking) contextual.
* **Saída:** Resposta fundamentada com citação precisa da fonte normativa.

---

## 2. Pipeline de Engenharia de Dados (ETL)

### Passo 2.1: Ingestão e Conversão Estruturada

Não utilize extratores de texto linear simples. Documentos normativos dependem de layout visual para denotar hierarquia.

* **Ação:** Implementar um módulo de *Layout-Aware Parsing* (ex: modelos de visão computacional aplicados a documentos).
* 
**Objetivo:** Converter arquivos PDF (incluindo digitalizados via OCR) para formato **Markdown**, preservando tabelas, cabeçalhos e a estrutura visual.



### Passo 2.2: Sanitização e Detecção de Revogação

Normas antigas podem ter sido revogadas. O sistema deve garantir segurança jurídica.

* **Lógica de Filtro:** Implementar regex ou análise de texto para identificar marcadores de revogação (ex: "Revogado pela Resolução X").
* **Metadados:** Marcar documentos ou trechos com tag `status: revogado` ou `status: vigente`.
* 
**Output:** Objeto JSON contendo o texto limpo em Markdown e metadados de vigência.

---

## 3. Estratégia de Indexação (Chunking Semântico-Hierárquico)

Esta é a etapa crítica para documentos legais. Não utilizar *chunking* por contagem fixa de caracteres, pois isso quebra a relação entre regra (Caput) e exceção (Parágrafos).

### Passo 3.1: Parsing Baseado em Estrutura Legal

O algoritmo deve identificar delimitadores textuais padronizados em normas:

* **Caput (Artigo):** Identificado por padrões como `Art. 1º`, `Art. 12`.
* **Parágrafos:** Identificados por `§`, `Parágrafo único`.
* **Incisos/Alíneas:** Identificados por algarismos romanos (`I`, `II`) ou letras (`a)`, `b)`).

### Passo 3.2: Agrupamento Lógico (Logical Grouping)

* **Regra de Agrupamento:** Um *chunk* (unidade de vetorização) deve conter o **Artigo (Caput)** + todos os seus **Parágrafos** e **Incisos** associados.
* 
**Justificativa:** Garante que a exceção nunca seja separada da regra principal, preservando a unidade semântica do dispositivo legal.



### Passo 3.3: Tratamento de Overflow e Recursividade

Para artigos excessivamente longos que excedam a janela de contexto do modelo de *embedding*:

* **Estratégia:** Dividir os parágrafos em *chunks* menores.
* **Contexto Herdado:** Obrigatoriamente, cada *chunk* "filho" (parágrafo isolado) deve receber uma cópia do texto do "pai" (Caput do artigo) como prefixo. Isso mantém a inteligibilidade vetorial do fragmento isolado.



### Passo 3.4: Enriquecimento de Metadados

Cada *chunk* deve ser um objeto JSON contendo:

* `content`: Texto do dispositivo legal completo.
* `metadata`:
* `hierarchy`: ["Título I", "Capítulo II", "Seção IV"] (Rastreabilidade).
* `source`: Nome do arquivo original.
* `category`: Tipo de norma (Estatuto, Resolução).
* `status`: Vigente/Revogado.



---

## 4. Pipeline de Recuperação (Retrieval Funnel)

Implementar uma estratégia de funil para maximizar a revocação (encontrar tudo o que é relevante) e a precisão (entregar apenas o útil).

### Estágio 1: Pré-Filtragem (Hard Filtering)

* Aplicar filtros nos metadados antes da busca vetorial.
* 
**Regra:** Excluir documentos com `status: revogado` para evitar alucinações jurídicas.



### Estágio 2: Busca Densa (Bi-Encoder)

* Utilizar um banco de dados vetorial (Vector DB) com algoritmo de busca aproximada (ex: HNSW).
* 
**Configuração:** Recuperar um número maior de candidatos (ex: `top_k = 50`) para garantir que a resposta correta esteja no conjunto, priorizando o *Recall*.



### Estágio 3: Reordenação (Cross-Encoder Reranking)

* Utilizar um modelo **Cross-Encoder** (que processa a Query e o Documento simultaneamente) para reavaliar os 50 candidatos.
* **Objetivo:** Capturar nuances sintáticas finas que a busca vetorial simples perde.
* 
**Seleção Final:** Selecionar apenas os `top_k = 5` documentos mais relevantes após a reordenação para enviar ao LLM.



---

## 5. Geração e Interface

### Passo 5.1: Construção do Prompt

* Injetar os 5 *chunks* recuperados no contexto do LLM.
* **Instrução de Sistema:** "Responda baseando-se estritamente no contexto fornecido. Cite o Artigo e a Norma de onde a informação foi extraída."

### Passo 5.2: Interface do Usuário

* A resposta deve exibir links ou referências diretas aos documentos originais, permitindo auditoria humana.



---

## 6. Protocolo de Avaliação (RAGAS)

Para validar a qualidade do sistema sem depender exclusivamente de humanos, utilizar o paradigma **LLM-as-a-Judge** (ex: Framework RAGAS).

### Métricas Obrigatórias:

1. **Faithfulness (Fidelidade):** A resposta deriva logicamente do contexto recuperado? (Mitigação de alucinação) .


2. 
**Answer Relevance:** A resposta atende à dúvida do usuário?.


3. **Context Precision:** Os documentos relevantes apareceram no topo do ranking?
4. 
**Context Recall:** O sistema encontrou toda a informação necessária para responder?.



### Dataset de Validação:

* Criar um conjunto híbrido: Perguntas sintéticas geradas por LLM baseadas nos documentos + Perguntas "Padrão-Ouro" (Golden Set) curadas manualmente por especialistas.