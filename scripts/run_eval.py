#!/usr/bin/env python3
"""
Script para executar a avaliação RAGAS.

Uso:
    python scripts/run_eval.py

Executa o pipeline completo (busca + rerank + geração) para cada pergunta
do golden dataset e calcula métricas RAGAS.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.evaluation.ragas_eval import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("AVALIAÇÃO RAGAS — Sistema RAG UNIVASF")
    logger.info("=" * 60)

    # Define caminho do relatório com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / "data" / f"ragas_report_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_evaluation(output_path=output_path)

    logger.info(f"\nRelatório salvo em: {output_path}")


if __name__ == "__main__":
    main()
