import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os
import time
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of stock symbols to train on
STOCKS = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",  # IT Sector
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS",  # Banking
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",  # Energy
    "SUNPHARMA.NS", "DRREDDY.NS",  # Pharma
    "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS",  # Automobiles
    "ITC.NS", "HINDUNILVR.NS",  # FMCG
    "TATASTEEL.NS"  # Metals
]

SEQ_LENGTH = 10  # Number of past days as input
DAYS = 90  # Fetch last 90 days of data

# Function to fetch stock data with retry mechanism
def fetch_stock_data(stock_symbol, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            data = yf.download(stock_symbol, period="3mo", interval="1d", timeout=10) # Added timeout
            data = data.ffill().bfill().dropna() # improved missing data handling
            if data is None or data.empty or len(data["Close"].dropna()) < SEQ_LENGTH:
                logging.warning(f"Insufficient data for {stock_symbol}, skipping...")
                return None
            return np.nan_to_num(data["Close"].dropna().values)
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries reached. Unable to fetch data for {stock_symbol}.")
                return None
        except Exception as e:
            logging.error(f"Error fetching stock data for {stock_symbol}: {e}")
            return None

# Initialize data storage
all_data, labels, scalers = [], [], {}

# Fetch and prepare data for multiple stocks
for stock in STOCKS:
    closing_prices = fetch_stock_data(stock)
    if closing_prices is None:
        continue  # Skip stock if data is insufficient

    # Normalize stock prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_prices = scaler.fit_transform(closing_prices.reshape(-1, 1))
    scalers[stock] = scaler  # Store scaler

    # Create input sequences
    def create_sequences(data, seq_length=SEQ_LENGTH):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X_stock, y_stock = create_sequences(normalized_prices)

    if len(X_stock) == 0:  # Prevent training on empty data
        logging.warning(f"Not enough data for {stock}, skipping...")
        continue

    all_data.append(X_stock)
    labels.append(y_stock)

if not all_data:
    logging.error("No valid stock data found for training!")
    sys.exit(1)

# Combine all stocks' training data
X_train = np.vstack(all_data)
y_train = np.concatenate(labels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

# Build LSTM model with Monte Carlo Dropout
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),  # Enable dropout for uncertainty estimation
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='linear')  # Ensure proper output scaling
])

model.compile(optimizer='adam', loss='mse')

# Train the model with early stopping
logging.info("Training LSTM model on multiple stocks...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #added early stopping
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1, callbacks=[early_stopping]) #added callbacks

# Save model and scalers
os.makedirs("models", exist_ok=True)
model.save("models/lstm_stock_model.h5")
joblib.dump(scalers, "models/scalers.pkl")

logging.info("Model and scalers saved successfully!")