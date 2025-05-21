import yfinance as yf
from datetime import datetime, timedelta
import json

def get_last_60_prices(symbol='IBM'):
    """
    Baixa os últimos 60 preços de fechamento da ação especificada.
    
    Retorna:
        Um dicionário no formato {"prices": [lista de 60 preços]}
    """
    end_date = datetime.today() - timedelta(days=1)  # ontem
    start_date = end_date - timedelta(days=100)  # margem para garantir 60 dias úteis

    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    closing_prices = df['Close'].dropna().values[-60:].flatten()  # <- AQUI: garante lista simples

    if len(closing_prices) < 60:
        raise ValueError(f"Dados insuficientes. Apenas {len(closing_prices)} preços foram obtidos.")

    return {"prices": closing_prices.tolist()}

if __name__ == "__main__":
    try:
        body = get_last_60_prices()
        json_body = json.dumps(body, indent=4)
        print("🔢 Body para API /predict:")
        print(json_body)
    except Exception as e:
        print(f"Erro: {e}")
