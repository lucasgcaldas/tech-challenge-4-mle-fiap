import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.utils.preprocessing import load_and_prepare_data
import os

# Carregar dados
X_train, y_train, X_test, y_test, scaler = load_and_prepare_data()

# Definir arquitetura do modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[es])

# Fazer predições
y_pred = model.predict(X_test)

# Inverter escala para métricas reais
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Calcular métricas
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print(f"\nAvaliação do Modelo:")
print(f"MAE  (Erro Absoluto Médio):        {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático):    {rmse:.2f}")
print(f"MAPE (Erro Percentual Médio):      {mape:.2f}%")

# Salvar o modelo
os.makedirs("app/model", exist_ok=True)
model.save("app/model/lstm_model.h5")
print("\n✅ Modelo salvo em: app/model/lstm_model.h5")
