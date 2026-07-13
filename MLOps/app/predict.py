#!/usr/bin/env python
"""
predict.py — Carrega o modelo treinado e calcula a probabilidade de fraude de uma linha.

Reaproveitado pelo main.py (Streamlit). O threshold de decisão vem de config.json,
não fica fixo no código — assim dá pra ajustar sem mexer no script.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "Model" / "fraud_model.pkl"
CONFIG_PATH = Path(__file__).with_name("config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    APP_CFG = json.load(fh)

logger.info("Carregando modelo: %s", MODEL_PATH)
with MODEL_PATH.open("rb") as fh:
    MODEL = pickle.load(fh)


def risk_level(probability: float) -> str:
    bands = APP_CFG["risk_bands"]
    if probability < bands["low_max"]:
        return "BAIXO"
    if probability < bands["medium_max"]:
        return "MEDIO"
    return "ALTO"


def predict(row_df: pd.DataFrame) -> dict:
    """Recebe uma linha no formato da ABT (462 colunas) e retorna a predição."""
    row_df = row_df[MODEL.feature_names_in_]  # garante mesma ordem/colunas do treino

    probability = float(MODEL.predict_proba(row_df)[:, 1][0])
    threshold = APP_CFG["threshold"]

    return {
        "fraud_probability": probability,
        "is_fraud": int(probability >= threshold),
        "risk_level": risk_level(probability),
        "threshold_used": threshold,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe várias linhas no formato da ABT e retorna uma predição por linha."""
    df = df.drop(columns=["isFraud"], errors="ignore")
    df = df[MODEL.feature_names_in_]  # garante mesma ordem/colunas do treino

    probabilities = MODEL.predict_proba(df)[:, 1]
    threshold = APP_CFG["threshold"]

    return pd.DataFrame({
        "fraud_probability": probabilities,
        "is_fraud": (probabilities >= threshold).astype(int),
        "risk_level": [risk_level(p) for p in probabilities],
    })
