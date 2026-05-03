import yfinance as yf

def save_stock_data(ticker, start_date, end_date):
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save to CSV
    filename = f"{ticker}_history.csv"
    data.to_csv(filename)
    print(f"Data for {ticker} saved to {filename}")

# Example usage
save_stock_data("AAPL", "2023-01-01", "2024-01-01")
