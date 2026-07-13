#!/usr/bin/env python
"""
build_reference_row.py — Gera os perfis-base e as estatísticas de escala usados pelo app.

O app de demonstração (main.py) só expõe alguns campos amigáveis no formulário
(valor da transação, produto, cartão, dispositivo). As outras ~450 colunas da
ABT (V1-V339, agregações, etc. — onde estão os maiores preditores do modelo,
variáveis Vesta opacas) precisam vir de algum lugar para o modelo receber uma
linha completa.

Testei usar a mediana de cada coluna (isFraud=0 e isFraud=1 separadamente), mas
como as colunas são independentes entre si, a mediana perde a correlação conjunta
que o modelo aprende — o "perfil de fraude" sintético não passava de ~15% de
probabilidade, mesmo sendo a mediana real das fraudes. Por isso usei, em vez
disso, duas TRANSAÇÕES REAIS da ABT como perfil-base, escolhidas por um critério
de NEGÓCIO (não pelo próprio modelo, para não escolher o exemplo com base na
opinião do modelo sobre ele mesmo):
- reference_row.json:              transação legítima com valor (TransactionAmt)
  mais próximo da mediana das transações legítimas — um caso "típico".
- reference_row_fraud_pattern.json: transação fraudulenta do produto C (cashout),
  o hotspot de fraude identificado na Etapa 1/2 (11,7% de fraude, ~4x a média),
  com valor mais próximo da mediana das fraudes desse produto.

Também salva scaling_stats.json: como o abt_transform.py aplica StandardScaler
em TODAS as colunas (inclusive flags binárias como M4_missing), os campos do
formulário precisam ser escalados da mesma forma antes de sobrescrever o perfil-base.
Para as flags binárias, guardo os dois valores escalados (0 e 1) já observados
na ABT. Para a contínua (TransactionAmt_log), guardo média/desvio padrão da
amostra para padronizar do mesmo jeito.

Roda uma única vez (os resultados já estão versionados). Uso:
    python MLOps/app/build_reference_row.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ABT_PATH = Path(__file__).parent.parent.parent / "Dados" / "abt.csv"
MODEL_PATH = Path(__file__).parent.parent.parent / "Model" / "fraud_model.pkl"
OUTPUT_TYPICAL = Path(__file__).with_name("reference_row.json")
OUTPUT_FRAUD = Path(__file__).with_name("reference_row_fraud_pattern.json")
OUTPUT_SCALING = Path(__file__).with_name("scaling_stats.json")
SAMPLE_ROWS = 150_000  # amostra suficiente; abt.csv tem 5GB+

# Colunas que o preprocessing.py sobrescreve a partir do formulário
BINARY_COLS = [
    "ProductCD_is_cashout", "card6_is_credit", "card4_is_discover",
    "DeviceType_is_mobile", "DeviceType_is_desktop", "DeviceType_missing",
    "M4_missing",
]
CONTINUOUS_COLS = ["TransactionAmt_log"]


def main() -> None:
    logger.info("Lendo amostra de %s linhas de %s", SAMPLE_ROWS, ABT_PATH)
    df = pd.read_csv(ABT_PATH, nrows=SAMPLE_ROWS)

    logger.info("Carregando modelo: %s", MODEL_PATH)
    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)

    # --- Estatísticas de escala (StandardScaler foi aplicado em tudo na ABT) ---
    scaling_stats: dict = {"binary": {}, "continuous": {}}
    for col in BINARY_COLS:
        scaling_stats["binary"][col] = {"0": float(df[col].min()), "1": float(df[col].max())}
    for col in CONTINUOUS_COLS:
        scaling_stats["continuous"][col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}

    with OUTPUT_SCALING.open("w", encoding="utf-8") as fh:
        json.dump(scaling_stats, fh, indent=2)
    logger.info("Estatísticas de escala salvas: %s", OUTPUT_SCALING)

    # --- Perfis-base: transações reais, escolhidas por critério de negócio (ver docstring acima) ---
    legit_df = df[df["isFraud"] == 0].reset_index(drop=True)
    fraud_df = df[df["isFraud"] == 1].reset_index(drop=True)

    cashout_flag_value = fraud_df["ProductCD_is_cashout"].max()  # valor escalado do flag=1
    fraud_cashout_df = fraud_df[fraud_df["ProductCD_is_cashout"] == cashout_flag_value]
    if fraud_cashout_df.empty:  # fallback de segurança (não deve ocorrer com a amostra padrão)
        fraud_cashout_df = fraud_df

    idx_legit = int((legit_df["TransactionAmt_log"] - legit_df["TransactionAmt_log"].median()).abs().argmin())
    idx_fraud = int((fraud_cashout_df["TransactionAmt_log"] - fraud_cashout_df["TransactionAmt_log"].median()).abs().argmin())

    legit_example = legit_df.drop(columns=["isFraud"]).iloc[idx_legit]
    fraud_example = fraud_cashout_df.drop(columns=["isFraud"]).iloc[idx_fraud]

    with OUTPUT_TYPICAL.open("w", encoding="utf-8") as fh:
        json.dump(legit_example.to_dict(), fh, indent=2)
    with OUTPUT_FRAUD.open("w", encoding="utf-8") as fh:
        json.dump(fraud_example.to_dict(), fh, indent=2)

    # Prob. prevista é só para log/documentação — não influencia a escolha acima.
    legit_prob = float(model.predict_proba(legit_example.to_frame().T[model.feature_names_in_])[:, 1][0])
    fraud_prob = float(model.predict_proba(fraud_example.to_frame().T[model.feature_names_in_])[:, 1][0])
    logger.info("Perfil típico salvo: %s (prob. prevista = %.4f)", OUTPUT_TYPICAL, legit_prob)
    logger.info("Perfil de fraude (cashout) salvo: %s (prob. prevista = %.4f)", OUTPUT_FRAUD, fraud_prob)


if __name__ == "__main__":
    main()
