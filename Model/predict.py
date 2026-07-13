#!/usr/bin/env python
"""
predict.py — Predição em lote de fraude para novas transações.

Espera um CSV já no formato da ABT do grupo (mesmas colunas de Dados/abt.csv,
geradas por DataPipeline/abt_transform.py). Carrega o modelo treinado
(fraud_model.pkl) e escreve um CSV de saída com a probabilidade de fraude e a
classificação de cada linha.

Uso:
    python Model/predict.py --input Dados/abt.csv --output predicoes.csv
    python Model/predict.py --input novas_transacoes.csv --output predicoes.csv --threshold 0.4
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).with_name("fraud_model.pkl")


def load_model(model_path: Path):
    logger.info("Carregando modelo: %s", model_path)
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def predict_batch(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aplica o modelo em um DataFrame já no formato da ABT."""
    df = df.drop(columns=["isFraud"], errors="ignore")
    X = df[model.feature_names_in_]

    logger.info("Calculando probabilidades para %s linhas...", len(X))
    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["fraud_probability"] = probabilities
    result["isFraud_pred"] = (probabilities >= threshold).astype(int)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predição em lote de fraude.")
    parser.add_argument("--input", required=True, help="CSV de entrada (formato ABT)")
    parser.add_argument("--output", required=True, help="CSV de saída com as predições")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Caminho do modelo treinado")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de decisão (default 0.5)")
    args = parser.parse_args()

    model = load_model(Path(args.model))

    logger.info("Carregando dados: %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Dados carregados: %s linhas x %s colunas", f"{len(df):,}", df.shape[1])

    result = predict_batch(model, df, args.threshold)

    output_path = Path(args.output)
    result.to_csv(output_path, index=False)

    fraud_count = int(result["isFraud_pred"].sum())
    logger.info(
        "Predições concluídas: %s fraudes previstas de %s (%.2f%%) | threshold=%.2f",
        f"{fraud_count:,}", f"{len(result):,}", fraud_count / len(result) * 100, args.threshold,
    )
    logger.info("Predições salvas: %s", output_path)


if __name__ == "__main__":
    main()
