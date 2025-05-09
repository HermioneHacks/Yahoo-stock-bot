import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed stock data :)
df = pd.read_csv("data/processed_stock_data.csv", index_col="Date", parse_dates=True)

# Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Moving Average Convergence Divergence (MACD)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Normalize Data
scaler = MinMaxScaler()
df[['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal Line']] = scaler.fit_transform(
    df[['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal Line']]
)

# **Fix: Fill missing values to prevent training errors**
df.fillna(method="bfill", inplace=True)  # Backfill to replace NaNs

# Save processed data
df.to_csv("data/processed_stock_data.csv")

print("✅ Preprocessed stock data saved with filled missing values!")
