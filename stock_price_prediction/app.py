import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify, make_response
from tensorflow.keras.losses import MeanSquaredError
import logging
import requests
from requests.exceptions import Timeout, ConnectionError
import time
from concurrent.futures import ThreadPoolExecutor  # For non-blocking yfinance calls

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model("models/lstm_stock_model.h5", custom_objects={"mse": MeanSquaredError})
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None

# Load scalers
try:
    scaler_data = joblib.load("models/scalers.pkl")
    if not isinstance(scaler_data, dict):
        raise ValueError("Scalers.pkl should contain a dictionary mapping stock symbols to scalers.")
except Exception as e:
    logging.error(f"Scaler loading failed: {e}")
    scaler_data = {}

STOCKS = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",  # IT Sector
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS",  # Banking
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",  # Energy
    "SUNPHARMA.NS", "DRREDDY.NS",  # Pharma
    "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS",  # Automobiles
    "ITC.NS", "HINDUNILVR.NS",  # FMCG
    "TATASTEEL.NS"  # Metals
]

# Use a ThreadPoolExecutor for non-blocking yfinance calls
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed

def fetch_stock_data(stock_symbol, max_retries=3, retry_delay=5):
    """Fetches stock data with retry logic and non-blocking execution."""
    for attempt in range(max_retries):
        try:
            # Use a future to make the yf.download call non-blocking
            future = executor.submit(yf.download, stock_symbol, period="2mo", interval="1d", timeout=10)
            df = future.result(timeout=20)  # Add a timeout to the future
            df = df.ffill().bfill()
            if df is None or df.empty:
                logging.warning(f"No historical price data available for {stock_symbol}")
                return None
            return df
        except (Timeout, ConnectionError) as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries reached. Unable to fetch data for {stock_symbol}.")
                return None
        except Exception as e:
            logging.error(f"Error fetching stock data for {stock_symbol}: {e}")
            return None

def get_initial_data():
    """Fetches initial data for the first load of the page."""
    initial_stock = STOCKS[0]
    df = fetch_stock_data(initial_stock)
    if df is not None and "Close" in df.columns and len(df) >= 30:
        dates = df.index.strftime("%Y-%m-%d").tolist()[-30:]
        closing_prices = df['Close'].dropna().values.tolist()[-30:]
        return dates, closing_prices, initial_stock
    return [], [], initial_stock

@app.route('/')
def index():
    """Renders the index page."""
    initial_dates, initial_prices, initial_stock = get_initial_data()
    return render_template('index.html', stocks=STOCKS, historical_prices=initial_prices, historical_dates=initial_dates, selected_stock=initial_stock)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles stock price prediction requests."""
    try:
        data = request.get_json()
        if not isinstance(data, dict) or 'stock_symbol' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        stock_symbol = data['stock_symbol']

        if stock_symbol not in STOCKS:
            return jsonify({"error": f"Stock {stock_symbol} is not supported."}), 400

        scaler = scaler_data.get(stock_symbol, None)
        if scaler is None:
            return jsonify({"error": f"No scaler found for {stock_symbol}. Ensure scalers.pkl has all required stocks."}), 400

        df = fetch_stock_data(stock_symbol)
        if df is None:
            return jsonify({"error": f"Failed to fetch stock data for {stock_symbol}"}), 500

        if "Close" not in df.columns:
            return jsonify({"error": f"'Close' column missing for {stock_symbol}"}), 400

        close_prices = df['Close'].dropna().values.flatten()

        if len(close_prices) < 10:
            return jsonify({"error": "Not enough valid closing price data to make a prediction"}), 400

        close_prices_scaled = scaler.transform(close_prices[-30:].reshape(-1, 1))

        X_test = np.array([close_prices_scaled[-10:]])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        def monte_carlo_predictions(model, X_input, n_simulations=100):
            f_preds = []
            for _ in range(n_simulations):
                prediction = model(X_input, training=True)
                f_preds.append(prediction)
            return np.array(f_preds)

        if model is not None:
            preds = monte_carlo_predictions(model, X_test)
            predicted_prices = scaler.inverse_transform(preds.reshape(-1, 1))
            mean_pred = np.mean(predicted_prices)
            std_pred = np.std(predicted_prices)

            predicted_price = round(float(mean_pred), 2) if mean_pred is not None else 0.0
            confidence_interval = min(max(round(float(std_pred / (predicted_price + 1e-8)), 2) * 100, 0), 100) if predicted_price else 0.0

            dates = df.index.strftime("%Y-%m-%d").tolist()[-30:]
            closing_prices = close_prices.tolist()[-30:]

            logging.info(
                f"Returning JSON: {{predicted_price: {predicted_price}, confidence_interval: {confidence_interval},"
                f" dates: {dates}, closing_prices: {closing_prices}}}")

            return jsonify({
                "predicted_price": predicted_price,
                "confidence_interval": confidence_interval,
                "dates": dates,
                "closing_prices": closing_prices
            })
        else:
            return jsonify({"error": "Model not loaded."}), 500

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(500)  # Catch internal server errors
def handle_500(error):
    """Handles 500 errors."""
    logging.error(f"Internal Server Error: {error}")
    return make_response(jsonify({"error": "Internal Server Error"}), 500)

@app.errorhandler(Exception)  # Catch all other exceptions
def handle_exception(error):
    """Handles all unhandled exceptions."""
    logging.error(f"Unhandled Exception: {error}")
    return make_response(jsonify({"error": "Internal Server Error"}), 500)

if __name__ == '__main__':
    app.run(debug=True, port = 5001)  # Use threaded=False with ThreadPoolExecutor
