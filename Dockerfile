FROM python:3.10-slim

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Primeiro copiar apenas os requisitos para cache
COPY requirements.txt .

# Instalar dependências do Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante da aplicação
COPY . .

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app
ENV PROMETHEUS_MULTIPROC_DIR=/tmp

# Porta exposta
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]