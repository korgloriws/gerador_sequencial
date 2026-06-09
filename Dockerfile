FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

# Prophet e dependências numéricas precisam de compiladores no build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# PyTorch CPU (imagem menor; suficiente para o LSTM do agente)
RUN pip install --upgrade pip && \
    pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements-docker.txt

COPY pattern_agent.py web_interface.py ./

RUN mkdir -p /app/uploads /app/data

EXPOSE 5000

# Treino/predição podem demorar vários minutos
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "900", \
     "--keep-alive", "30", \
     "web_interface:app"]
