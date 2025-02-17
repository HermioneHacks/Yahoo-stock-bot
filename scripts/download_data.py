import yfinance as yf
import pandas as pd
import os

# Define stock ticker and time range
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2024-01-01"

# Fetch stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)

# Reset index to ensure 'Date' is a proper column
stock_data.reset_index(inplace=True)

# Drop the extra row by specifying the correct columns to keep
correct_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
# Drop the first row if it does not contain a date
stock_data = stock_data[stock_data['Date'].str.contains(r"\d{4}-\d{2}-\d{2}", na=False)]
stock_data = stock_data[correct_columns]

# Save raw stock data with correct structure
stock_data.to_csv("data/raw_stock_data.csv", index=False)

print(f"✅ Downloaded stock data for {ticker} and saved to data/raw_stock_data.csv correctly")
