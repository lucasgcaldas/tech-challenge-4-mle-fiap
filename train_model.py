import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import L1L2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from app.utils.preprocessing import load_and_prepare_data
import os

# Carregar dados
X_train, y_train, X_test, y_test, scaler = load_and_prepare_data()

# Arquitetura aprimorada do modelo
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Otimizador com learning rate ajustável
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Treinamento com validação
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[es, mc],
    verbose=1
)

# Plotar histórico de treino
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Fazer predições
y_pred = model.predict(X_test)

# Inverter escala
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Métricas aprimoradas
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return mae, rmse, mape, r2

mae, rmse, mape, r2 = calculate_metrics(y_test_inv, y_pred_inv)

print("\n📊 Avaliação do Modelo:")
print(f"MAE  (Erro Absoluto Médio):        {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático):    {rmse:.2f}")
print(f"MAPE (Erro Percentual Médio):      {mape:.2f}%")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

# Salvar modelo e scaler
os.makedirs("app/model", exist_ok=True)
model.save("app/model/lstm_model.h5")
np.save("app/model/scaler_params.npy", np.array([scaler.min_, scaler.scale_]))

print("\n✅ Modelo e scaler salvos em:")
print("- app/model/lstm_model.h5")
print("- app/model/scaler_params.npy")
