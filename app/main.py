from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Summary
import uvicorn
import os
import logging
import time

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização da API
app = FastAPI(
    title="Stock Price Predictor - IBM",
    description="API para previsão de preços de ações utilizando modelo LSTM",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app)

# Métricas Prometheus
PREDICTION_TIME = Summary("prediction_processing_seconds", "Tempo de processamento da predição")
PREDICTION_COUNT = Counter("prediction_total", "Total de requisições de predição")
MODEL_MAE = Gauge("model_mae", "Erro Absoluto Médio da última predição")
MODEL_RMSE = Gauge("model_rmse", "Raiz do Erro Quadrático Médio da última predição")
MODEL_MAPE = Gauge("model_mape", "Erro Percentual Médio da última predição")
MODEL_R2 = Gauge("model_r2", "R² da última predição")
MODEL_LOADED = Gauge("model_loaded", "Indica se o modelo foi carregado corretamente")

# Constantes de caminho
MODEL_DIR = "app/model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.npy")

# Carregar modelo e scaler

def load_model_and_scaler():
    try:
        custom_objects = {
            'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
            'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError()
        }

        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='mse')

        scaler_params = np.load(SCALER_PATH)
        target_scaler = MinMaxScaler()
        target_scaler.min_, target_scaler.scale_ = scaler_params[0], scaler_params[1]

        MODEL_LOADED.set(1)
        logger.info("Modelo e scaler carregados com sucesso")
        return model, target_scaler
    except Exception as e:
        logger.error(f"Erro ao carregar modelo ou scaler: {str(e)}")
        raise RuntimeError("Falha ao inicializar o serviço")

model, target_scaler = load_model_and_scaler()

# Schemas da API
class PriceInput(BaseModel):
    prices: List[float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "prices": [150.2, 151.5, 152.3, 149.8, 150.7] * 12
            }
        }
    }

class PredictionOutput(BaseModel):
    predicted_price: float
    confidence: float
    status: str

# Função de predição
@PREDICTION_TIME.time()
def make_prediction(prices: List[float]) -> PredictionOutput:
    PREDICTION_COUNT.inc()

    if len(prices) != 60:
        raise HTTPException(status_code=400, detail="Você deve fornecer exatamente 60 preços de fechamento.")

    prices_np = np.array(prices).reshape(-1, 1)
    prices_scaled = target_scaler.transform(prices_np)
    X_input = np.repeat(prices_scaled, 9, axis=1).reshape(1, 60, 9)
    pred_scaled = model.predict(X_input)
    pred_original = target_scaler.inverse_transform(pred_scaled)

    true_price = prices[-1]
    predicted_price = float(pred_original[0][0])

    mae = abs(true_price - predicted_price)
    rmse = np.sqrt((true_price - predicted_price) ** 2)
    mape = abs((true_price - predicted_price) / true_price) * 100
    r2 = 1 - ((true_price - predicted_price)**2 / ((true_price - np.mean(prices))**2 + 1e-8))

    MODEL_MAE.set(mae)
    MODEL_RMSE.set(rmse)
    MODEL_MAPE.set(mape)
    MODEL_R2.set(r2)

    volatility = np.std(prices) / np.mean(prices)
    confidence = max(0, 1 - volatility) * 100

    return PredictionOutput(
        predicted_price=round(predicted_price, 2),
        confidence=round(confidence, 2),
        status="success"
    )

# Endpoints
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PriceInput):
    try:
        return make_prediction(input_data.prices)
    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à API de Previsão de Ações IBM",
        "endpoints": {
            "/predict": "POST - Envie os últimos 60 preços para previsão",
            "/docs": "Documentação da API",
            "/metrics": "Métricas do Prometheus"
        }
    }

# Execução local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)