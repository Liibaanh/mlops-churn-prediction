FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]