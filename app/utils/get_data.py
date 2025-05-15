import yfinance as yf
from datetime import datetime

def download_ibm_data(start='2018-01-01', path='data/stock_data.csv'):
    symbol = 'IBM'
    df = yf.download(symbol, start=start, end=datetime.now())
    df.to_csv(path)
    print(f'Dados salvos em: {path}')

if __name__ == "__main__":
    download_ibm_data()