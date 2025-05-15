FROM python:3.10-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar os arquivos do projeto
COPY . .

# Instalar dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando para rodar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
