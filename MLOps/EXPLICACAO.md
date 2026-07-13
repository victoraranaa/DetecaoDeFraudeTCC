# Etapa 6 — Deployment & MLOps (Deploy e Operação)

## O que é essa etapa?

> Esta etapa é **individual** — cada membro do grupo entrega sua própria implementação, em cima do mesmo modelo treinado pelo grupo (`Model/fraud_model.pkl`). Nada do modelo, da ABT ou dos scripts de `DataPipeline/`/`Model/train.py` é alterado aqui.

---

## O que foi implementado

Este projeto implementa **só o app Streamlit** — não há API REST (FastAPI) nem banco de dados de logs. Não existe endpoint `/predict` neste projeto.

```
raw (Dados/raw/*.csv)
      │  DataPipeline/data_sanitization.py   (grupo)
      ▼
Dados/clean_data.csv ──────► copiado para Dados/silver/clean_data.csv
      │  DataPipeline/abt_transform.py       (grupo)
      ▼
Dados/abt.csv ──────────────► copiado para Dados/gold/abt.csv ─────────┐
      │  Model/predict.py (lote, individual)                           │
      ▼                                                                 │
predicoes.csv                                                           │
                                                                         │
MLOps/app/  (individual, tempo real)                               │
  ├─ main.py            Streamlit — formulário + resultado
  ├─ preprocessing.py    monta a linha de 462 colunas
  ├─ predict.py           carrega o modelo, aplica o threshold
  ├─ reference_row*.json  perfis-base (transações reais da ABT) ◄───────┘
  ├─ scaling_stats.json   estatísticas p/ escalar os campos do formulário
  └─ config.json          threshold de decisão

MLOps/pipeline_config.json                  caminhos das camadas/scripts
```

## Componentes do Deploy

### 1. Dashboard Streamlit (`MLOps/app/main.py`)

O app tem duas abas:
- **Predição Individual**: formulário com 6 campos editáveis (valor da transação, produto, bandeira e tipo de cartão, dispositivo, verificação M4) → `preprocessing.py` monta a linha completa de 462 colunas → `predict.py` consulta o modelo → a tela mostra probabilidade, classificação (FRAUDE/LEGÍTIMA) e nível de risco (BAIXO/MÉDIO/ALTO).
- **Predição em Lote (CSV)**: upload de um CSV já no formato da ABT → `predict.predict_batch()` roda o modelo em todas as linhas de uma vez → tabela com probabilidade/classificação/risco por linha + botão de download do resultado.

Abaixo das abas, o app também mostra a **performance do modelo** (ROC-AUC, Precisão, Recall, F1-Score e matriz de confusão, lidos de `Model/training_metrics.json`) e o gráfico de importância das principais features.

O modelo tem 462 colunas porque foi treinado com todas as variáveis Vesta (V1-V339, opacas — a Vesta não divulga o significado). Não faz sentido pedir isso num formulário, então as ~450 colunas que o usuário não edita vêm de uma **transação real** da base de treino (ver "Perfis-base" abaixo) — isso vale só para a aba individual; a aba de lote espera um CSV que já tem as 462 colunas prontas.

### 2. Predição em lote (`Model/predict.py`)

Script que processa um CSV inteiro de uma vez (já no formato da ABT):
```
Input: Dados/abt.csv (ou outro CSV com as mesmas colunas)
  ↓
Modelo carregado (fraud_model.pkl)
  ↓
Output: predicoes.csv com colunas fraud_probability e isFraud_pred
```

É um script separado do `MLOps/app/predict.py`, mas as duas versões de lote fazem a mesma conta (carregam o modelo, aplicam o threshold em várias linhas de uma vez): esta aqui é pra rodar pelo terminal (sem abrir o app); a do app (`predict_batch()`, usada pela aba "Predição em Lote") é pra quem prefere fazer o upload do CSV pela tela em vez da linha de comando. `MLOps/app/predict.py` também tem `predict()`, que prevê uma linha por vez — essa sim é exclusiva do formulário individual do Streamlit.

### 3. Perfis-base (`MLOps/app/build_reference_row.py`)

Como preencher as ~450 colunas que o formulário não edita? Uma coluna sozinha não diz nada sobre fraude — o que o modelo aprendeu foi a combinação entre elas. Por isso, montar o perfil-base coluna por coluna (por exemplo, pegando a mediana de cada uma) não funciona: o resultado é uma linha "artificial" que não corresponde a nenhuma transação de verdade, e o modelo trata isso como algo fora do padrão que ele conhece (na prática, testei e o "perfil de fraude" saía com probabilidade prevista bem baixa, mesmo usando medianas de fraudes reais).

A solução foi usar **duas transações reais** da ABT como ponto de partida — assim a combinação entre as colunas continua sendo uma combinação que realmente aconteceu:
- **Típica**: transação legítima com valor mais próximo da mediana das legítimas.
- **Padrão de fraude**: transação fraudulenta do produto C (cashout — o hotspot de 11,7% de fraude identificado na Etapa 1/2), com valor mais próximo da mediana das fraudes desse produto.

Repara que o critério de escolha das duas transações (valor da transação e produto) não depende em nada do que o modelo prevê para elas — se eu escolhesse o exemplo perguntando pro próprio modelo "qual dessas transações você classificaria melhor?", estaria usando a resposta pra validar a pergunta.

### 4. Containerização com Docker

O app é empacotado em um container Docker para rodar em qualquer máquina sem precisar instalar Python/dependências manualmente:

```dockerfile
FROM python:3.11-slim
COPY MLOps/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Model/fraud_model.pkl Model/fraud_model.pkl
COPY MLOps/app ./MLOps/app
CMD ["streamlit", "run", "MLOps/app/main.py", ...]
```

`docker-compose.yml` só tem **um serviço** (`app`) — não há banco de dados nem serviço separado de pipeline, porque o objetivo aqui é demonstrar o modelo, não simular uma infraestrutura de produção completa.

```yaml
services:
  app:
    build: { context: .., dockerfile: MLOps/Dockerfile }
    ports: ["8501:8501"]
```

### 5. Pipeline orquestrado (`MLOps/pipeline_orchestration.py`)

Roda em sequência os 3 passos do fluxo (dado bruto → limpo → ABT → predição em lote), cada um como um processo separado, os mesmos scripts do grupo sem alterá-los. Se uma etapa falhar, o pipeline para — não faz sentido gerar a ABT em cima de dados sujos. Depois das duas primeiras etapas, o script também copia o resultado para a pasta da camada correspondente (ver item 6).

### 6. Arquitetura em camadas (raw/silver/gold)

As camadas seguem a **arquitetura medalhão** (raw/silver/gold):

| Camada | O que é | Neste projeto |
|--------|---------|----------------|
| **Raw** | Dado bruto, como chegou da fonte, sem tratamento | `Dados/raw/*.csv` (CSVs originais do Kaggle) |
| **Silver** | Dado limpo/validado, mas ainda "cru" em termos de negócio | `Dados/silver/clean_data.csv` (cópia de `Dados/clean_data.csv`, saída da sanitização) |
| **Gold** | Dado pronto para consumo (relatório, modelo, dashboard) | `Dados/gold/abt.csv` (cópia da ABT, o que o modelo usa) |

As pastas `silver` e `gold` são **cópias organizadas**, criadas pelo próprio `pipeline_orchestration.py` (função `organize_layer`, ver item 5) depois de cada etapa. Os arquivos originais (`Dados/clean_data.csv`, `Dados/abt.csv`) continuam nos mesmos lugares de sempre, porque o resto do projeto (scripts do grupo e o app da Etapa 6) já espera esses caminhos. Criar cópias evita ter que mexer no código do grupo só para renomear pastas. Os nomes das camadas vêm de `MLOps/pipeline_config.json`, não estão fixos no código.

**Por que não usei um orquestrador dedicado (Airflow):** é a ferramenta mais comum para esse tipo de pipeline e aparece no diagrama de referência do PDF do projeto, mas orquestrar de verdade (agendamento, retentativas, interface de monitoramento) é um estudo novo em cima de tudo que já falta pra entrega. Preferi ter uma arquitetura em camadas funcional e simples de explicar dado o prazo.

### 7. Página de referência sobre pré-processamento (`MLOps/app/pages/1_Pre_processamento.py`)

O Streamlit tem uma convenção simples de múltiplas páginas: qualquer arquivo `.py` dentro de uma pasta `pages/`, ao lado do script principal, vira automaticamente um item no menu lateral — não precisa escrever nenhuma configuração de navegação.

Essa página não recalcula nada — só lê dois arquivos que o grupo já gera (`Dados/clean_data_report.json` e `Dados/abt_metadata.json`) e mostra:
- **Raw → Silver:** quantas linhas/colunas antes e depois da limpeza, distribuição de fraude vs. legítima, outliers detectados por coluna, variáveis categóricas com mais valores ausentes.
- **Silver → Gold:** quantas features vieram de cada grupo (V, C, D, M, cartão, etc.), a lista de transformações que o `abt_transform.py` aplicou, e as variáveis mais correlacionadas com fraude.

---

## MLOps: Monitoramento do Modelo

Modelos de fraude se degradam com o tempo — os padrões de fraude mudam e o perfil de transações também.

### Data Drift
```
Treino: TransactionAmt médio = USD 134
Produção após 3 meses: TransactionAmt médio = USD 189
→ Modelo pode estar "desatualizado"
```

### Model Drift (Concept Drift)
Novos tipos de fraude que o modelo nunca viu no treino mudam a relação entre as features e o target.

### Métricas a Monitorar em Produção

| Métrica | Frequência | Alerta se... |
|---------|-----------|-------------|
| Taxa de fraudes detectadas | Diária | Cai > 5 pontos percentuais |
| Taxa de falsos positivos | Diária | Sobe > 2 pontos percentuais |
| Distribuição de TransactionAmt | Semanal | KS test p-value < 0.05 (data drift) |
| AUC-ROC (em amostra rotulada) | Mensal | Cai abaixo de 0,85 |

**Gatilho de retreino:** se o AUC-ROC cair abaixo de 0,85 numa amostra rotulada mensal, ou se o KS test indicar drift significativo em `TransactionAmt`, retreinar com dados mais recentes usando o mesmo `Model/train.py` do grupo.

---

## Como Rodar Localmente

```bash
# App via Docker
cd MLOps
docker compose up --build
# abrir http://localhost:8501

# App sem Docker
pip install -r MLOps/requirements.txt
streamlit run MLOps/app/main.py

# Predição em lote
python Model/predict.py --input Dados/abt.csv --output predicoes.csv

# Pipeline completo (bruto -> ABT -> predição), já organiza Dados/silver/ e Dados/gold/
python MLOps/pipeline_orchestration.py --output predicoes.csv
```
