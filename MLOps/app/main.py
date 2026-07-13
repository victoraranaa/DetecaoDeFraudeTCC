#!/usr/bin/env python
"""
main.py — Dashboard Streamlit de demonstração do modelo de detecção de fraudes.

Formulário com os campos de uma transação -> preprocessing.py monta a linha
completa -> predict.py consulta o modelo -> mostra o risco na tela.

Uso:
    streamlit run MLOps/app/main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import predict
import preprocessing

TRAINING_METRICS_PATH = Path(__file__).parent.parent.parent / "Model" / "training_metrics.json"

st.set_page_config(page_title="Detecção de Fraudes — Banco FIA MBA Digital", page_icon="🔍")

st.title("Detecção de Fraudes em Transações")
st.caption("Banco FIA MBA Digital · Etapa 6 — Deploy (demonstração individual)")

st.info(
    "Este app é uma **simplificação para demonstração**. O modelo real usa 462 "
    "variáveis (incluindo ~339 variáveis Vesta opacas, V1-V339). Aqui só os campos "
    "abaixo são editáveis; as demais colunas vêm de uma transação real da base de "
    "treino, escolhida no seletor **Perfil-base**."
)

tab_individual, tab_lote = st.tabs(["Predição Individual", "Predição em Lote (CSV)"])

with tab_individual:
    with st.form("transacao_form"):
        st.subheader("Perfil-base")
        base_profile_label = st.radio(
            "Ponto de partida das ~450 colunas não editáveis",
            options=["Transação típica (legítima)", "Transação com padrão de fraude conhecido"],
            help=(
                "Como os maiores preditores do modelo são variáveis Vesta opacas (não "
                "editáveis aqui), o perfil-base 'padrão de fraude' usa uma transação "
                "fraudulenta real do histórico para que os campos abaixo tenham mais "
                "efeito visível sobre o risco."
            ),
        )
        base_profile = "fraud_pattern" if "fraude" in base_profile_label else "tipico"

        st.subheader("Dados da Transação")
        col1, col2 = st.columns(2)

        with col1:
            transaction_amt = st.number_input("Valor da transação (USD)", min_value=1.0, max_value=20000.0, value=150.0)
            product_cd = st.selectbox(
                "Produto",
                options=["W", "C", "H", "R", "S"],
                format_func=lambda p: {
                    "W": "W — Web (2,0% fraude)",
                    "C": "C — Cashout (11,7% fraude — hotspot)",
                    "H": "H — Hotel/Home (4,8% fraude)",
                    "R": "R — Recorrente (3,8% fraude)",
                    "S": "S — Serviços (5,9% fraude)",
                }[p],
            )
            card4 = st.selectbox("Bandeira do cartão", options=["visa", "mastercard", "discover", "amex"])
            card6 = st.selectbox("Tipo de cartão", options=["debit", "credit"], format_func=lambda c: "Débito" if c == "debit" else "Crédito")

        with col2:
            device_type = st.selectbox(
                "Dispositivo",
                options=["missing", "desktop", "mobile"],
                format_func=lambda d: {"missing": "Não informado", "desktop": "Desktop", "mobile": "Mobile"}[d],
            )
            m4_ausente = st.checkbox(
                "Verificação M4 ausente/falhou",
                help="Maior preditor do modelo depois das variáveis Vesta — indica que uma verificação de segurança não pôde ser concluída.",
            )

        submitted = st.form_submit_button("Calcular risco de fraude")

    if submitted:
        row = preprocessing.build_input_row(
            transaction_amt=transaction_amt,
            product_cd=product_cd,
            card4=card4,
            card6=card6,
            device_type=device_type,
            m4_ausente=m4_ausente,
            base_profile=base_profile,
        )
        result = predict.predict(row)

        st.subheader("Resultado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilidade de fraude", f"{result['fraud_probability']*100:.1f}%")
        col2.metric("Classificação", "FRAUDE" if result["is_fraud"] else "LEGÍTIMA")
        col3.metric("Nível de risco", result["risk_level"])
        st.progress(min(result["fraud_probability"], 1.0))
        st.caption(f"Threshold de decisão usado: {result['threshold_used']} (ver MLOps/app/config.json)")

with tab_lote:
    st.subheader("Predição em Lote")
    st.caption(
        "Envie um CSV no mesmo formato da ABT (Dados/abt.csv, 462 colunas geradas por "
        "DataPipeline/abt_transform.py) — o mesmo formato aceito por Model/predict.py."
    )

    uploaded_file = st.file_uploader("Arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write(f"{len(input_df):,} linhas carregadas.")

        if st.button("Rodar predição em lote"):
            result_df = predict.predict_batch(input_df)

            n_fraude = int(result_df["is_fraud"].sum())
            st.write(f"**{n_fraude:,}** transações classificadas como fraude de **{len(result_df):,}** no total.")

            st.subheader("Resultado")
            st.dataframe(result_df, hide_index=True)

            csv_saida = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar resultado (CSV)",
                data=csv_saida,
                file_name="predicoes.csv",
                mime="text/csv",
            )

st.divider()

with TRAINING_METRICS_PATH.open("r", encoding="utf-8") as fh:
    metrics = json.load(fh)
test_metrics = metrics["test_metrics"]

st.subheader("Performance do Modelo (base de teste)")
st.caption("Métricas do XGBoost treinado — ver Model/train.py e Model/training_metrics.json.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
col2.metric("Precisão", f"{test_metrics['precision']*100:.1f}%")
col3.metric("Recall", f"{test_metrics['recall']*100:.1f}%")
col4.metric("F1-Score", f"{test_metrics['f1']*100:.1f}%")

st.caption("Matriz de confusão (base de teste)")
matriz_confusao = pd.DataFrame(
    [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]],
    index=["Real: Legítima", "Real: Fraude"],
    columns=["Previsto: Legítima", "Previsto: Fraude"],
)
st.dataframe(matriz_confusao)

st.divider()
st.subheader("Top Features do Modelo (referência)")
st.caption("Importância global do XGBoost treinado — a maioria são variáveis Vesta (V1-V339), proprietárias e não editáveis neste formulário.")

top_features = pd.DataFrame(test_metrics["feature_importance_top20"][:8])
st.bar_chart(top_features.set_index("feature"))
