#!/usr/bin/env python
"""
1_Pre_processamento.py — Página de referência sobre a limpeza dos dados (raw -> silver)
e a construção da ABT (silver -> gold).

Só lê os metadados que os scripts do grupo já geram (Dados/clean_data_report.json e
Dados/abt_metadata.json) — não recalcula nada e não abre os CSVs (o maior, abt.csv,
tem 5,5GB, não faz sentido carregar isso num app de demonstração).

Aparece automaticamente no menu lateral do Streamlit por estar na pasta pages/,
ao lado de main.py — não precisa de nenhuma configuração extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # pages/ -> app/ -> MLOps/ -> raiz
CLEAN_REPORT_PATH = PROJECT_ROOT / "Dados" / "clean_data_report.json"
ABT_METADATA_PATH = PROJECT_ROOT / "Dados" / "abt_metadata.json"

st.set_page_config(page_title="Pré-processamento — Banco FIA MBA Digital", page_icon="🧹")

st.title("Pré-processamento dos Dados")
st.caption(
    "Página de referência sobre as Etapas 2 e 3 (grupo) — não recalcula nada, só mostra "
    "os metadados já gerados por DataPipeline/data_sanitization.py e abt_transform.py."
)

with CLEAN_REPORT_PATH.open("r", encoding="utf-8") as fh:
    clean_report = json.load(fh)

with ABT_METADATA_PATH.open("r", encoding="utf-8") as fh:
    abt_meta = json.load(fh)

st.header("1. Raw → Silver: Limpeza dos dados")
st.caption("`DataPipeline/data_sanitization.py`")

col1, col2, col3 = st.columns(3)
col1.metric("Linhas", f"{clean_report['final_shape'][0]:,}")
col2.metric("Colunas", clean_report["final_shape"][1])
col3.metric("Linhas removidas", f"{clean_report['rows_removed_pct']:.1f}%")

st.caption(
    f"Shape inicial {clean_report['initial_shape'][0]:,} × {clean_report['initial_shape'][1]} → "
    f"final {clean_report['final_shape'][0]:,} × {clean_report['final_shape'][1]}. Nenhuma linha "
    "removida nesta etapa — os outliers abaixo foram só identificados aqui; o tratamento "
    "(ex.: `TransactionAmt_log`) acontece na construção da ABT."
)

fraude = clean_report["target_distribution"]["fraude"]
legitimo = clean_report["target_distribution"]["legitimo"]
st.write(
    f"**Distribuição do target:** {legitimo:,} legítimas / {fraude:,} fraudes "
    f"({clean_report['target_fraud_rate_pct']:.2f}% de fraude)"
)

st.subheader("Outliers detectados (método IQR)")
outliers_df = pd.DataFrame(clean_report["outliers"])
outliers_df = outliers_df[outliers_df["count"] > 0].sort_values("pct", ascending=False)
st.dataframe(
    outliers_df[["column", "count", "pct"]].rename(
        columns={"column": "Coluna", "count": "Qtde. outliers", "pct": "% da base"}
    ),
    hide_index=True,
)

st.subheader("Variáveis categóricas com mais valores ausentes")
cat_df = pd.DataFrame(clean_report["categorical_profiles"])
cat_df = cat_df.sort_values("missing_pct", ascending=False).head(10)
st.dataframe(
    cat_df[["name", "n_unique", "missing_pct"]].rename(
        columns={"name": "Variável", "n_unique": "Valores únicos", "missing_pct": "% ausente"}
    ),
    hide_index=True,
)

st.divider()

st.header("2. Silver → Gold: Construção da ABT")
st.caption("`DataPipeline/abt_transform.py`")

col1, col2 = st.columns(2)
col1.metric("Features finais", abt_meta["dataset_info"]["total_features"])
col2.metric(
    "Treino / Teste",
    f"{abt_meta['dataset_info']['train_rows']:,} / {abt_meta['dataset_info']['test_rows']:,}",
)

st.subheader("De onde vêm as 462 features")
groups_df = pd.DataFrame(
    list(abt_meta["feature_groups"].items()), columns=["grupo", "quantidade"]
).sort_values("quantidade", ascending=False)
st.bar_chart(groups_df.set_index("grupo"))

st.subheader("Principais transformações aplicadas")
for transformacao in abt_meta["transformations_applied"]:
    st.markdown(f"- {transformacao}")

st.subheader("Variáveis mais correlacionadas com fraude (referência da EDA)")
strong_df = pd.DataFrame(abt_meta["strong_predictors"]).head(10)
st.bar_chart(strong_df.set_index("feature"))

st.divider()
st.caption(
    "Depois de rodar `MLOps/pipeline_orchestration.py`, essas mesmas camadas também "
    "existem fisicamente em `Dados/raw/`, `Dados/silver/` e `Dados/gold/` (ver README da Etapa 6)."
)
