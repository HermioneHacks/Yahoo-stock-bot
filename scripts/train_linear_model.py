import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the preprocessed stock data
df = pd.read_csv("data/processed_stock_data.csv", index_col="Date", parse_dates=True)

# Define Features (X) and Target Variable (Y)
features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal Line']
X = df[features]
y = df['Close']

# Split into training (80%) and testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label="Actual Price", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Price", color='red', linestyle='dashed')
plt.title("Linear Regression - Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Normalized Stock Price")
plt.legend()
plt.show()
