from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import os

# Inicializar FastAPI
app = FastAPI(title="Stock Price Predictor - IBM")

# Instrumentar a API com Prometheus
Instrumentator().instrument(app).expose(app)

# Carregar o modelo treinado
MODEL_PATH = "app/model/lstm_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Criar classe de entrada
class PriceInput(BaseModel):
    prices: list[float]  # Últimos 60 preços

@app.post("/predict")
def predict(input_data: PriceInput):
    prices = input_data.prices

    if len(prices) != 60:
        return {"error": "Você deve fornecer exatamente 60 preços de fechamento."}

    # Normalizar os dados com MinMaxScaler
    scaler = MinMaxScaler()
    prices_np = np.array(prices).reshape(-1, 1)
    scaled = scaler.fit_transform(prices_np)

    # Preparar entrada para o modelo
    X_input = scaled.reshape(1, 60, 1)
    pred_scaled = model.predict(X_input)

    # Reverter normalização (usando os mesmos dados de escala)
    pred_original = scaler.inverse_transform(pred_scaled)

    return {"predicted_price": round(float(pred_original[0][0]), 2)}

@app.get("/")
def root():
    return {"message": "API de Previsão de Ações IBM - Use /predict para enviar dados."}
