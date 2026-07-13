# Etapa 6 — Deploy e MLOps (individual)

Banco FIA MBA Digital · Detecção de Fraudes · CRISP-DM Etapa 6

Esta pasta contém a produtização do modelo treinado pelo grupo
(`Model/fraud_model.pkl`, XGBoost). É entrega **individual** — não altera o
modelo, a ABT ou os scripts de `DataPipeline/`/`Model/train.py` do grupo.

## Arquitetura

O pipeline segue a arquitetura em camadas (raw → silver → gold):

```
raw (Dados/raw/*.csv)
      │  DataPipeline/data_sanitization.py  (grupo)
      ▼
Dados/clean_data.csv ──────► copiado para Dados/silver/clean_data.csv
      │  DataPipeline/abt_transform.py      (grupo)
      ▼
Dados/abt.csv ──────────────► copiado para Dados/gold/abt.csv ────────┐
      │  Model/predict.py (lote, individual)                          │
      ▼                                                                │
predicoes.csv                                                          │
                                                                        │
MLOps/app/  (individual, tempo real)                               │
  ├─ main.py            Streamlit — formulário + resultado
  ├─ preprocessing.py    monta a linha de 462 colunas
  ├─ predict.py           carrega o modelo, aplica o threshold
  ├─ reference_row*.json  perfis-base (transações reais da ABT) ◄──────┘
  ├─ scaling_stats.json   estatísticas p/ escalar os campos do formulário
  ├─ config.json          threshold de decisão
  └─ pages/1_Pre_processamento.py   página extra: limpeza + ABT (Etapas 2/3)

MLOps/pipeline_config.json        caminhos das camadas/scripts
MLOps/pipeline_orchestration.py    roda as 3 etapas acima em sequência e organiza
                                        as cópias em Dados/silver/ e Dados/gold/
MLOps/Dockerfile + docker-compose.yml  empacota o app Streamlit
```

**Sobre as camadas raw/silver/gold:** `Dados/silver/clean_data.csv` e `Dados/gold/abt.csv` são cópias organizadas, geradas pelo próprio `pipeline_orchestration.py` depois de cada etapa. Os arquivos originais (`Dados/clean_data.csv`, `Dados/abt.csv`) continuam nos caminhos de sempre, porque o resto do projeto (scripts do grupo e o app) já espera esses caminhos — criar cópias evita ter que alterar o código do grupo só para reorganizar pastas.

## O que o app mostra

O app (`MLOps/app/main.py`) tem duas abas:
- **Predição Individual**: formulário de 6 campos → resultado (probabilidade, classificação, nível de risco).
- **Predição em Lote (CSV)**: upload de um CSV no formato da ABT → tabela com o resultado de todas as linhas + download.

Abaixo das abas, uma seção fixa mostra a **performance do modelo** (ROC-AUC, Precisão, Recall, F1, matriz de confusão) e o gráfico de importância das principais features — dados lidos de `Model/training_metrics.json`, sem recalcular nada.

## Por que o app simplifica a entrada (aba Individual)

O modelo usa 462 colunas (incluindo ~339 variáveis Vesta opacas, V1-V339, que
são os maiores preditores — ver `Model/evaluation.ipynb`). Não faz sentido pedir isso
num formulário. A aba Individual expõe só 6 campos interpretáveis (valor, produto,
bandeira/tipo de cartão, dispositivo, verificação M4) e preenche o resto com o
valor de uma **transação real** da ABT de treino — o usuário escolhe entre um
perfil "típico" (legítimo) e um "padrão de fraude conhecido" (cashout). Essa
simplificação não existe na aba de Lote: lá o CSV enviado já precisa ter as
462 colunas prontas (mesmo formato que `Model/predict.py` espera).

Testei usar a mediana de cada coluna em vez de uma transação real, mas a
mediana por coluna quebra a correlação entre variáveis que o modelo aprendeu —
o "perfil de fraude" sintético não passava de ~15% de probabilidade. Uma
transação real do histórico resolve isso.

As duas transações-base não são escolhidas pelo próprio modelo (isso seria
circular — usar o modelo para escolher o exemplo que o demonstra). O critério
é de negócio: a transação real com valor (`TransactionAmt`) mais próximo da
mediana da sua classe — para a fraude, dentro do produto C (cashout), o
hotspot de 11,7% de fraude já identificado na Etapa 1/2. Detalhes em
`MLOps/app/build_reference_row.py`.

**Hora da transação não é um campo do formulário.** O `abt_transform.py` do
grupo codifica a hora como seno/cosseno (para hora 23 e hora 0 ficarem
"próximas" matematicamente) — não incluí isso no formulário porque hora não
está entre as features mais importantes do modelo (ver
`Model/training_metrics.json`) e adicionaria complexidade desnecessária pro
usuário. Essas colunas (e as flags de risco por horário) ficam com o valor
original do perfil-base, como as outras ~450 colunas não editáveis.

**Detalhe importante notado ao construir isso:** o `StandardScaler` do `abt_transform.py` do
grupo é aplicado em todas as colunas, inclusive flags binárias (`M4_missing`,
`ProductCD_is_cashout`, etc.) — por isso `MLOps/app/preprocessing.py` escala
os campos do formulário com as mesmas estatísticas antes de montar a linha.
Também notei que `IsWeekend` é uma feature constante na ABT (o cálculo de
`HourOfDay_risk`/`IsWeekend` em `abt_transform.py` faz `HourOfDay // 24` ser
sempre 0) — mais um motivo para não expor nada ligado a horário no formulário.

## Como rodar

### App interativo (Docker)
```bash
cd MLOps
docker compose up --build
# abrir http://localhost:8501
```

### App interativo (sem Docker)
```bash
pip install -r MLOps/requirements.txt
streamlit run MLOps/app/main.py
```

### Predição em lote
```bash
python Model/predict.py --input Dados/abt.csv --output predicoes.csv --threshold 0.5
```

### Pipeline completo (bruto → ABT → predição)
```bash
python MLOps/pipeline_orchestration.py --output predicoes.csv
# ou, se Dados/clean_data.csv e Dados/abt.csv já existem:
python MLOps/pipeline_orchestration.py --skip-sanitization --skip-abt --output predicoes.csv
```

## Página extra: Pré-processamento (`MLOps/app/pages/1_Pre_processamento.py`)

A professora pediu para mostrar um pouco do pré-processamento. Como o Streamlit cria menu lateral automaticamente para qualquer arquivo dentro de `MLOps/app/pages/`, essa página aparece sozinha ao lado da principal — sem nenhuma configuração extra.

Ela só lê os metadados que o grupo já gera (`Dados/clean_data_report.json` e `Dados/abt_metadata.json`) — não recalcula nada e não abre os CSVs (o `abt.csv` sozinho tem 5,5GB, não faz sentido carregar isso num app de demonstração):
- **Raw → Silver:** shape antes/depois da limpeza, distribuição do target, outliers detectados por coluna (método IQR), variáveis categóricas com mais valores ausentes.
- **Silver → Gold:** total de features por grupo (V, C, D, M, cartão, engenharia, flags de ausência), lista das transformações aplicadas, variáveis mais correlacionadas com fraude.

## Threshold de decisão

O threshold usado (`MLOps/app/config.json` e `--threshold` do `predict.py`)
está em **0,5**, igual ao `Model/train.py` do grupo.

## Monitoramento em Produção (MLOps)

Modelos de fraude se degradam com o tempo — padrões de fraude mudam e o perfil
de transações também. O que monitorar:

| Métrica | Frequência | Alerta se... |
|---------|-----------|-------------|
| Taxa de fraudes detectadas | Diária | Cai > 5 pontos percentuais |
| Taxa de falsos positivos | Diária | Sobe > 2 pontos percentuais |
| Distribuição de TransactionAmt | Semanal | KS test p-value < 0.05 (data drift) |
| AUC-ROC (em amostra rotulada) | Mensal | Cai abaixo de 0,85 |

- **Data drift:** a distribuição das variáveis de entrada muda (ex.: ticket
  médio subindo de USD 134 para USD 189) — o modelo passa a extrapolar fora da
  faixa em que foi treinado.
- **Model/concept drift:** a relação entre as variáveis e a fraude muda (novos
  padrões de fraude surgem que o modelo nunca viu).
- **Gatilho de retreino:** se AUC-ROC cair abaixo de 0,85 em uma amostra
  rotulada mensal, ou se o KS test indicar drift significativo em
  `TransactionAmt`/`V_top20_sum`, retreinar com dados mais recentes usando o
  mesmo `Model/train.py` do grupo.

## Próximos Passos — Ações Automatizadas (ML + Automação + Agentes de IA)

O app hoje só mostra o resultado na tela — quem decide o que fazer com aquilo ainda é uma pessoa. O passo natural depois dessa entrega é conectar a predição a uma ação automática, sem esperar alguém olhar o dashboard. Como o app já calcula um nível de risco (BAIXO/MÉDIO/ALTO, ver `config.json`), cada faixa pode disparar uma ação diferente:

- **BAIXO risco:** aprovar direto, sem intervenção humana — é a maioria das transações e não faz sentido gerar atrito nelas.
- **MÉDIO risco:** cair numa fila de revisão manual (política de negócio prevista desde a Etapa 1 — probabilidade entre 0,3 e 0,7). Hoje essa fila seria só uma lista; o próximo passo é ter um **agente de IA** que, antes de a transação chegar ao analista, já monta um resumo automático: qual foi o valor, o produto, o histórico recente daquele cartão e quais features pesaram mais na decisão do modelo (usando a importância de features que já tenho em `training_metrics.json`). Isso poupa o analista de abrir várias telas para entender o caso.
- **ALTO risco:** bloquear a transação automaticamente e disparar uma notificação (SMS/e-mail) pro portador confirmar se foi ele mesmo — um fluxo parecido com o que os bancos já fazem quando o cartão é usado num lugar incomum.

Do lado do monitoramento (seção anterior), a ideia é parecida: em vez de alguém checar o drift manualmente todo mês, um agente poderia rodar essa checagem sozinho e, se encontrar uma queda de AUC-ROC ou um drift significativo, abrir automaticamente um chamado pro time de dados avaliar o retreino — a pessoa só entra quando já existe um sinal concreto de que algo mudou, não para checar se mudou.

Nada disso está implementado neste projeto — é uma proposta de próximo passo, pensada pra mostrar que a solução tem para onde crescer depois desta entrega, e não pra ser confundida com algo que já está rodando.
