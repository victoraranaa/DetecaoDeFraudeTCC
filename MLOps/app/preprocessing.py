#!/usr/bin/env python
"""
preprocessing.py — Monta a linha de entrada do modelo a partir dos campos do formulário.

O modelo (Model/fraud_model.pkl) espera as mesmas 462 colunas da ABT (Dados/abt.csv),
geradas pelo DataPipeline/abt_transform.py do grupo — inclusive com StandardScaler
aplicado em TODAS as colunas (até flags binárias como M4_missing). Como o app de
demonstração não recebe uma transação bruta completa (com as ~339 variáveis V da
Vesta, por exemplo), ele simplifica a entrada:

- Os campos do formulário viram as poucas colunas derivadas que dependem deles
  diretamente (log do valor, flags de produto/cartão/dispositivo, M4), já
  escalados do mesmo jeito que o treino (ver scaling_stats.json).
- Todas as demais colunas vêm de um perfil-base: uma transação REAL da ABT,
  não uma média sintética. O usuário escolhe entre "tipico" (transação
  legítima real) e "fraud_pattern" (transação fraudulenta real, do hotspot
  cashout). O motivo de usar uma transação real (em vez de médias) e o
  critério de escolha das duas estão documentados em build_reference_row.py.

Isso é uma simplificação deliberada da Etapa 6 (individual) — não altera o modelo
nem a ABT do grupo, só a forma como o app monta a entrada para consultá-lo.

Observação: hora da transação não é um campo do formulário (motivo detalhado
no README da pasta) — essas colunas, e as demais ~450 não editáveis, ficam com
o valor original do perfil-base.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_ROW_TIPICO_PATH = Path(__file__).with_name("reference_row.json")
REFERENCE_ROW_FRAUD_PATH = Path(__file__).with_name("reference_row_fraud_pattern.json")
SCALING_STATS_PATH = Path(__file__).with_name("scaling_stats.json")

with REFERENCE_ROW_TIPICO_PATH.open("r", encoding="utf-8") as fh:
    REFERENCE_ROW_TIPICO = pd.Series(json.load(fh))

with REFERENCE_ROW_FRAUD_PATH.open("r", encoding="utf-8") as fh:
    REFERENCE_ROW_FRAUD = pd.Series(json.load(fh))

with SCALING_STATS_PATH.open("r", encoding="utf-8") as fh:
    _SCALING = json.load(fh)


def _scale_binary(col: str, flag: bool) -> float:
    """Retorna o valor já escalado (StandardScaler) correspondente a 0 ou 1."""
    return _SCALING["binary"][col]["1" if flag else "0"]


def _scale_continuous(col: str, raw_value: float) -> float:
    """Padroniza um valor contínuo com a média/desvio-padrão observados na ABT."""
    stats = _SCALING["continuous"][col]
    return (raw_value - stats["mean"]) / stats["std"]


def build_input_row(
    transaction_amt: float,
    product_cd: str,
    card4: str,
    card6: str,
    device_type: str,
    m4_ausente: bool,
    base_profile: str = "tipico",
) -> pd.DataFrame:
    """Monta uma linha de 462 colunas pronta para o modelo, a partir dos campos amigáveis.

    product_cd: um de "W", "C", "H", "R", "S" (C = cashout, hotspot de fraude)
    card4: um de "visa", "mastercard", "discover", "amex"
    card6: "credit" ou "debit"
    device_type: "desktop", "mobile" ou "missing" (sem dado de dispositivo)
    m4_ausente: se a verificação M4 não pôde ser feita (top preditor do modelo)
    base_profile: "tipico" ou "fraud_pattern" — perfil-base das colunas não editáveis
    """
    if base_profile == "fraud_pattern":
        row = REFERENCE_ROW_FRAUD.copy()
    else:
        row = REFERENCE_ROW_TIPICO.copy()

    row["TransactionAmt_log"] = _scale_continuous("TransactionAmt_log", np.log1p(transaction_amt))

    row["ProductCD_is_cashout"] = _scale_binary("ProductCD_is_cashout", product_cd == "C")
    row["card6_is_credit"] = _scale_binary("card6_is_credit", card6 == "credit")
    row["card4_is_discover"] = _scale_binary("card4_is_discover", card4 == "discover")

    row["DeviceType_is_mobile"] = _scale_binary("DeviceType_is_mobile", device_type == "mobile")
    row["DeviceType_is_desktop"] = _scale_binary("DeviceType_is_desktop", device_type == "desktop")
    row["DeviceType_missing"] = _scale_binary("DeviceType_missing", device_type == "missing")

    row["M4_missing"] = _scale_binary("M4_missing", m4_ausente)

    return pd.DataFrame([row])
