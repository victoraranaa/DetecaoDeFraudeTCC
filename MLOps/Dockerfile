# Dockerfile do app de demonstração (Etapa 6 — Deploy individual).
# Build a partir da raiz do projeto: docker-compose faz isso automaticamente
# (ver context em docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

COPY MLOps/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Model/fraud_model.pkl Model/fraud_model.pkl
COPY Model/training_metrics.json Model/training_metrics.json
COPY Dados/clean_data_report.json Dados/clean_data_report.json
COPY Dados/abt_metadata.json Dados/abt_metadata.json
COPY MLOps/app ./MLOps/app

EXPOSE 8501

CMD ["streamlit", "run", "MLOps/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
