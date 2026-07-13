# Detecção de Fraudes em Transações Financeiras

Projeto final do MBA Pós Big Data e Analytics (FIA Business School / LABDATA). O grupo escolheu o dataset IEEE-CIS Fraud Detection (Kaggle) para construir, do início ao fim, um modelo que identifica transações fraudulentas antes de elas serem aprovadas.

## O problema

Simulamos o caso de um banco fictício, o **Banco FIA MBA Digital**, que opera um gateway de pagamentos card-not-present (cartão não presente, típico de e-commerce), com atuação predominante nos EUA. No período coberto pelo dataset — 182 dias, cerca de 590 mil transações — 3,5% delas eram fraudulentas, o que representou USD 3,08 milhões em prejuízo sobre um volume total de USD 79,7 milhões movimentados. Quem sente esse problema não é só o banco: é o portador do cartão clonado, é o lojista que arca com o chargeback, e é o próprio banco que perde dinheiro e reputação a cada fraude que passa despercebida.

Um detalhe que direcionou boa parte das decisões do projeto: o produto "C" (cashout, saque via cartão) concentra 11,7% de fraude — quase 4 vezes a média geral. Esse hotspot aparece de novo mais adiante, tanto na análise exploratória quanto no app de demonstração da parte individual (`MLOps/`).

## Objetivo

Treinar um modelo de classificação binária que prevê `isFraud` (0 = legítima, 1 = fraude) a partir dos dados da transação e da identidade do portador, com metas de:

- **AUC-ROC > 0,90** (métrica principal, robusta ao desbalanceamento de 3,5% de fraudes)
- **Recall > 80%** (minimizar fraude que passa despercebida — é prejuízo direto)
- **Precision > 70%** (evitar bloquear cliente legítimo demais — é atrito e perda de receita)

O modelo final (XGBoost, com ajuste de hiperparâmetros via Optuna) chegou a ROC-AUC de 0,9643, Precision de 83,7% e Recall de 74,1% no conjunto de teste.

## Metodologia

Segui o CRISP-DM:

1. **Business Understanding** — definição do problema, das métricas de sucesso e das restrições operacionais (latência, política de revisão manual).
2. **Data Understanding (EDA)** — análise exploratória das 590.540 transações e 434 variáveis (depois do merge de `train_transaction` com `train_identity`), incluindo os principais hotspots de fraude e o mapeamento de valores ausentes. Ver `DataPipeline/exp_analysis.ipynb`.
3. **Data Preparation** — limpeza dos dados brutos e construção da ABT (Analytical Base Table) com 462 features, incluindo engenharia de variáveis específica para fraude (flags de ausência, agregações, codificação cíclica de horário, interações). Ver `DataPipeline/data_sanitization.py` e `DataPipeline/abt_transform.py`.
4. **Modeling** — comparação entre XGBoost, LightGBM e Random Forest, com ajuste fino de hiperparâmetros via Optuna e validação cruzada. Ver `Model/train.py`.
5. **Evaluation** — análise de métricas (AUC-ROC, Precision, Recall, matriz de confusão) e análise de erros. Ver `Model/evaluation.ipynb`.
6. **Deployment** — produtização do modelo num app de demonstração (Streamlit) com Docker, entregue individualmente por cada integrante do grupo. Ver `MLOps/`.

## Estrutura do repositório

```
Dados/                 Dados brutos, limpos e a ABT final (ver seção de treino abaixo)
DataPipeline/          Scripts de limpeza, construção da ABT e notebook de EDA
Model/                 Treino, predição em lote e avaliação do modelo
MLOps/                 Deploy individual (app Streamlit + Docker)
requirements.txt       Dependências do projeto inteiro
```

## Como treinar o modelo

Os dados brutos do Kaggle (`train_transaction.csv`, `train_identity.csv`, e os arquivos de teste) precisam estar em `Dados/raw/` antes de rodar qualquer coisa.

```bash
# 1. Instalar as dependências
pip install -r requirements.txt

# 2. Limpar os dados brutos
python DataPipeline/data_sanitization.py

# 3. Construir a ABT (feature engineering)
python DataPipeline/abt_transform.py

# 4. Treinar o modelo (compara XGBoost, LightGBM e Random Forest, depois ajusta hiperparâmetros com Optuna)
python Model/train.py

# Para pular o Optuna e treinar mais rápido com parâmetros padrão:
python Model/train.py --no-optuna
```

O treino gera `Model/fraud_model.pkl` (modelo final), `Model/training_metrics.json` (métricas completas) e `Model/optuna_study.pkl` (histórico da otimização). Para avaliar o modelo já treinado com mais profundidade — curvas ROC/PR, análise de threshold, explicabilidade — abra `Model/evaluation.ipynb`.

## A parte individual (deploy)

Cada integrante do grupo produtizou o modelo por conta própria, em cima do mesmo `fraud_model.pkl`. A minha versão está em `MLOps/`, com um app Streamlit onde dá pra simular uma transação e ver o risco calculado na hora, além do Docker Compose e do script que roda o pipeline completo de ponta a ponta. As instruções de como rodar estão no `README.md` daquela pasta.
