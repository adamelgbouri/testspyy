import yfinance as yf

portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

print("Daily Performance Tracker:")
print("-" * 30)

for ticker in portfolio:
    stock = yf.Ticker(ticker)
    # Get the last two days of data
    hist = stock.history(period="2d")
    
    if len(hist) < 2: continue
    
    prev_close = hist['Close'].iloc[0]
    curr_close = hist['Close'].iloc[1]
    change = ((curr_close - prev_close) / prev_close) * 100
    
    print(f"{ticker}: {change:+.2f}%")
