import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(path='data/stock_data.csv', sequence_length=60):
    df = pd.read_csv(path)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input

    # Dividir em treino e teste (80/20)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler
