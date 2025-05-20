from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import os
import logging
from typing import List

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Price Predictor - IBM",
    description="API para previsão de preços de ações utilizando modelo LSTM",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = "app/model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.npy")

try:
    # SOLUÇÃO ATUALIZADA PARA NOVAS VERSÕES DO TENSORFLOW/KERAS
    custom_objects = {
        'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
        'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError(),
        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError()
    }
    
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False  # Importante para evitar conflitos
    )
    
    # Compilar manualmente se necessário
    model.compile(optimizer='adam', loss='mse')
    
    scaler_params = np.load(SCALER_PATH)
    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler_params[0], scaler_params[1]
    
    logger.info("Modelo e scaler carregados com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar modelo ou scaler: {str(e)}")
    raise RuntimeError("Falha ao inicializar o serviço")

class PriceInput(BaseModel):
    prices: List[float]  # Últimos 60 preços
    model_config = {
        "json_schema_extra": {
            "example": {
                "prices": [150.2, 151.5, 152.3, ..., 149.8]  # 60 valores
            }
        }
    }

class PredictionOutput(BaseModel):
    predicted_price: float
    confidence: float  # Poderia ser a volatilidade ou outra métrica
    status: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PriceInput):
    """
    Faz a previsão do próximo preço com base nos últimos 60 valores
    
    Args:
        input_data: Lista com os últimos 60 preços de fechamento
    
    Returns:
        Objeto com preço previsto e metadados
    """
    try:
        prices = input_data.prices
        
        if len(prices) != 60:
            raise HTTPException(
                status_code=400,
                detail="Você deve fornecer exatamente 60 preços de fechamento."
            )
        
        # Converter para numpy array e normalizar
        prices_np = np.array(prices).reshape(-1, 1)
        prices_scaled = target_scaler.transform(prices_np)
        
        # Criar input com 9 features (repetindo o preço para todas as features)
        # Isso é uma solução temporária - o ideal seria ter todas as features originais
        X_input = np.repeat(prices_scaled, 9, axis=1)
        X_input = X_input.reshape(1, 60, 9)
        
        # Fazer previsão
        pred_scaled = model.predict(X_input)
        pred_original = target_scaler.inverse_transform(pred_scaled)
        
        # Calcular "confiança" baseada na volatilidade dos inputs
        volatility = np.std(prices) / np.mean(prices)
        confidence = max(0, 1 - volatility) * 100  # Convertendo para porcentagem
        
        return {
            "predicted_price": round(float(pred_original[0][0]), 2),
            "confidence": round(float(confidence), 2),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar a requisição: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da API"""
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.get("/")
async def root():
    """Endpoint raiz com informações básicas"""
    return {
        "message": "Bem-vindo à API de Previsão de Ações IBM",
        "endpoints": {
            "/predict": "POST - Envie os últimos 60 preços para receber uma previsão",
            "/docs": "Documentação interativa da API",
            "/metrics": "Métricas do Prometheus"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )