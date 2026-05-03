import yfinance as yf
import numpy as np

ticker = "NVDA"
data = yf.download(ticker, start="2023-01-01")['Close']

# Calculate daily percentage returns
returns = data.pct_change()

# Calculate annualized volatility (standard deviation * sqrt of trading days)
volatility = returns.std() * np.sqrt(252)

print(f"Annualized Volatility for {ticker}: {volatility:.2%}")
