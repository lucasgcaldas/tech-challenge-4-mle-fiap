import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path='data/stock_data.csv', sequence_length=60, test_size=0.2, random_state=42):
    """
    Carrega e prepara dados para modelo LSTM
    
    Parâmetros:
    - path: caminho para o arquivo CSV
    - sequence_length: tamanho da janela temporal
    - test_size: proporção dos dados para teste
    - random_state: seed para reprodutibilidade
    
    Retorna:
    - X_train, y_train: dados de treino
    - X_test, y_test: dados de teste
    - target_scaler: scaler ajustado apenas para o target
    """
    
    # 1. Carregar e preparar os dados
    df = pd.read_csv(path, skiprows=3)
    df.columns = ['Date', 'Price', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Engenharia de features
    df['Price_Change'] = df['Price'].pct_change()
    df['MA_7'] = df['Price'].rolling(window=7).mean()
    df['MA_21'] = df['Price'].rolling(window=21).mean()
    df['Volatility'] = df['Price'].rolling(window=7).std()
    df = df.dropna()
    
    # 3. Normalização separada para features e target
    features = df.drop(columns=['Date', 'Price'])
    target = df[['Price']]
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # 4. Combinar features e target normalizados
    scaled_data = np.column_stack((scaled_target, scaled_features))
    
    # 5. Criar sequências temporais
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :])  # Todas as features
        y.append(scaled_data[i, 0])  # Apenas o preço (target)
    
    X = np.array(X)
    y = np.array(y)
    
    # 6. Divisão treino-teste (preservando ordem temporal)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 7. Ajustar formato para LSTM (samples, timesteps, features)
    # Já está correto pelo método de construção
    
    return X_train, y_train, X_test, y_test, target_scaler