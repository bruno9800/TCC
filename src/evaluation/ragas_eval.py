"""
Pipeline de Avaliação RAGAS

Avalia a qualidade do sistema RAG usando o framework RAGAS
com métricas de Faithfulness, Answer Relevance, Context Precision e Context Recall.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    """Carrega o dataset de validação."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    search_engine: HybridSearchEngine | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Executa a avaliação RAGAS completa.

    Pipeline por pergunta:
    1. Busca híbrida → top-50 candidatos
    2. Reranking → top-5
    3. Geração → resposta com citações
    4. Avaliação RAGAS sobre o conjunto

    Args:
        search_engine: Motor de busca híbrida (cria se None).
        output_path: Onde salvar o relatório (opcional).

    Returns:
        Dicionário com scores por métrica.
    """
    if search_engine is None:
        search_engine = HybridSearchEngine()

    golden = load_golden_dataset()

    # Coleta de dados para o RAGAS
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(golden):
        query = item["question"]
        gt = item["ground_truth"]

        logger.info(f"[{i+1}/{len(golden)}] Processando: {query[:60]}...")

        # 1. Busca híbrida
        candidates = search_engine.search_hybrid(query)

        # 2. Reranking
        top_results = rerank(query, candidates)

        # 3. Geração
        generation = generate_answer(query, top_results)

        # Salva dados
        questions.append(query)
        answers.append(generation.answer)
        contexts.append([r.content for r in top_results])
        ground_truths.append(gt)

    # 4. Avaliação RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    report = {
        "metrics": {k: float(v) for k, v in results.items()},
        "num_questions": len(golden),
        "details": [
            {
                "question": questions[i],
                "answer": answers[i][:200] + "...",
                "num_contexts": len(contexts[i]),
                "ground_truth": ground_truths[i],
            }
            for i in range(len(golden))
        ],
    }

    # Salva relatório
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório salvo em: {output_path}")

    logger.info(f"\n{'='*60}")
    logger.info("RESULTADOS DA AVALIAÇÃO RAGAS:")
    for metric, value in report["metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")

    return report
