# tech-challenge-4-mle-fiap

## Integrantes Grupo 42

- Lucas Gomes - RM358850

## 📽️ Demonstração em Vídeo

📺 Vídeo explicativo: https://youtu.be/p6jcjaGJ3Zc

Modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.

# 📈 Stock Price Prediction API with LSTM

Este projeto utiliza redes neurais recorrentes (LSTM) para prever o **preço de fechamento das ações da IBM**, com base em séries temporais de dados históricos. O modelo é servido via **API RESTful com FastAPI** e está pronto para deploy em ambientes Docker.

---

## 🚀 Funcionalidades

- 📊 Coleta automática de dados históricos com `yfinance`
- 🔄 Pré-processamento e normalização com `scikit-learn`
- 🧠 Treinamento de modelo LSTM com `TensorFlow`
- 🔁 Avaliação com métricas: MAE, RMSE e MAPE
- 🌐 API RESTful para predição via HTTP POST
- 🐳 Deploy com Docker

---

## 📥 Instalação Local

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/stock-lstm-api.git
cd stock-lstm-api
````

### 2. Crie um ambiente virtual e instale as dependências

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

pip install -r requirements.txt
```

### 3. Baixe os dados e treine o modelo

```bash
python app/utils/get_data.py
python train_model.py
```

### 4. Rode a API localmente

```bash
uvicorn app.main:app --reload
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Docker

### Build da imagem

```bash
docker build -t ibm-stock-api .
```

### Rodar o container

```bash
docker run -d -p 8000:8000 ibm-stock-api
```

---

## 🔍 Exemplo de uso da API

### Endpoint: `/predict`

**Método:** `POST`
**Body JSON:**

```json
{
  "prices": [147.83, 148.01, ..., 148.12]  // exatamente 60 preços de fechamento
}
```

**Resposta:**

```json
{
  "predicted_price": 149.56
}
```

---

## 📊 Métricas do Modelo

* **MAE** (Erro Médio Absoluto): \~2.x
* **RMSE** (Raiz do Erro Quadrático Médio): \~3.x
* **MAPE** (Erro Percentual Absoluto Médio): \~1.x%

---

## 🧠 Tecnologias Utilizadas

* Python 3.10+
* TensorFlow / Keras
* yFinance
* FastAPI
* Docker

---

## Métricas

### 🔸 **V1 – Usando Bidirectional**

* **MAE**: 71.68
* **RMSE**: 80.24
* **MAPE**: 38.15%
* **R²**: -3.9519 ❌
  🔍 *Modelo com desempenho insatisfatório – erros muito altos.*

---

### 🔸 **V2**

* **MAE**: 6.26
* **RMSE**: 8.54
* **MAPE**: 3.33%
* **R²**: 0.9439 ✅
  🔧 *Boa melhora, porém ainda impreciso em alguns pontos.*

---

### 🔸 **V3**

* **MAE**: 3.14
* **RMSE**: 4.56
* **MAPE**: 1.71%
* **R²**: 0.9840 ✅
  ⚡ *Modelo mais preciso e consistente.*

---

### 🔸 **V4 (FINAL)**

* **MAE**: 4.01
* **RMSE**: 5.92
* **MAPE**: 2.17%
* **R²**: 0.9730 ✅
  🚀 *Modelo robusto e balanceado, escolhido para produção.*

---

## 📅 Resultados Diários

### 🔹 **20/05**

* **Previsto**: 264.74

---

### 🔹 **21/05**

* **Previsto**: 267.49
* **Real**: 268.41
* **Erro Absoluto**: |267.49 - 268.41| = **0.92**
* **Erro Percentual**: (0.92 / 268.41) × 100 = **0.34%**

---

### 🔹 **22/05**

* **Previsto**: 269.79
* **Real**: 260.87
* **Erro Absoluto**: |269.79 - 260.87| = **8.92**
* **Erro Percentual**: (8.92 / 260.87) × 100 = **3.42%**

---

### 🔹 **23/05**

* **Previsto**: 270.96
* **Real**: 258.37
* **Erro Absoluto**: |270.96 - 258.37| = **12.59**
* **Erro Percentual**: (12.59 / 258.37) × 100 = **4.87%**

---

### 🔹 **24/05**

* **Previsto**: 271.11
* **Real**: 258.63
* **Erro Absoluto**: |271.11 - 258.63| = **12.48**
* **Erro Percentual**: (12.48 / 258.63) × 100 = **4.82%**

---

### 🔹 **25/05**

* **Previsto**: 270.71
* **Real**: *não disponível*
* ✅ *Aguardando valor real*

---

## 📌 Resumo dos Erros Observados (21/05 a 24/05)

| Data  | Real   | Previsto | Erro Absoluto | Erro Percentual |
| ----- | ------ | -------- | ------------- | --------------- |
| 21/05 | 268.41 | 267.49   | 0.92          | 0.34%           |
| 22/05 | 260.87 | 269.79   | 8.92          | 3.42%           |
| 23/05 | 258.37 | 270.96   | 12.59         | 4.87%           |
| 24/05 | 258.63 | 271.11   | 12.48         | 4.82%           |

📈 **Média dos Erros Reais (21 a 24/05)**:

* **MAE real**: (0.92 + 8.92 + 12.59 + 12.48) / 4 = **8.23**
* **MAPE real**: (0.34 + 3.42 + 4.87 + 4.82) / 4 = **3.86%**

---

## 📌 Observação Final

➡️ Embora o modelo tenha um **MAPE médio de 2.17% nos dados de teste**, nos últimos dias os erros percentuais **aumentaram para uma média de 3.86%**, indicando uma possível mudança no comportamento do mercado ou a necessidade de **reajuste/atualização dos dados** de treino.
