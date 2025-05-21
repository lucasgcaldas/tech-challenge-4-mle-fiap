# tech-challenge-4-mle-fiap
Modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.


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

## 🏗️ Estrutura do Projeto

```

stock-lstm-api/
├── app/
│   ├── main.py               # Código da API
│   ├── model/
│   │   └── lstm\_model.h5     # Modelo treinado salvo
│   └── utils/
│       ├── get\_data.py       # Coleta de dados da IBM
│       └── preprocessing.py  # Funções de preparação de dados
├── data/
│   └── stock\_data.csv        # Dados históricos baixados
├── train\_model.py            # Treinamento do modelo LSTM
├── Dockerfile                # Imagem para container Docker
├── requirements.txt          # Dependências do projeto
└── README.md

````

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

## 📽️ Demonstração em Vídeo

📺 Assista ao vídeo explicativo: \[link\_YouTube\_ou\_Loom\_aqui]

---

## 📬 Contato

Projeto desenvolvido por \[Seu Nome].
📧 Email: [seu.email@exemplo.com](mailto:seu.email@exemplo.com)
🔗 LinkedIn: [linkedin.com/in/seu-usuario](https://linkedin.com/in/seu-usuario)


##

V1 (Usando Bidirectional)
📊 Avaliação do Modelo:
MAE  (Erro Absoluto Médio):        71.68
RMSE (Raiz do Erro Quadrático):    80.24
MAPE (Erro Percentual Médio):      38.15%
R²   (Coeficiente de Determinação): -3.9519

V2
📊 Avaliação do Modelo:
MAE  (Erro Absoluto Médio):        6.26
RMSE (Raiz do Erro Quadrático):    8.54
MAPE (Erro Percentual Médio):      3.33%
R²   (Coeficiente de Determinação): 0.9439

V3
📊 Avaliação do Modelo:
MAE  (Erro Absoluto Médio):        3.14
RMSE (Raiz do Erro Quadrático):    4.56
MAPE (Erro Percentual Médio):      1.71%
R²   (Coeficiente de Determinação): 0.9840

.
.
.

V4 (FINAL)
📊 Avaliação do Modelo:
MAE  (Erro Absoluto Médio):        4.01
RMSE (Raiz do Erro Quadrático):    5.92
MAPE (Erro Percentual Médio):      2.17%
R²   (Coeficiente de Determinação): 0.9730