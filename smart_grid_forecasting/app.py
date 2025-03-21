from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from db_config import get_db_connection
from datetime import datetime, timedelta

app = Flask(__name__)

# Load trained models
custom_objects = {"mse": MeanSquaredError()}
xgb_model = joblib.load("models/xgboost_model.pkl")
lstm_model = load_model("models/lstm_model.h5", custom_objects=custom_objects)
scaler = joblib.load("models/scaler.pkl")

# Load dataset
df = pd.read_csv("dataset/long_data_.csv")
df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors='coerce')
available_states = sorted(df["States"].unique().tolist())

@app.route("/")
def home():
    return render_template("dashboard.html", states=available_states)

@app.route("/get_states", methods=["GET"])
def get_states():
    return jsonify(available_states)

@app.route("/forecast", methods=["POST"])
def forecast():
    """Predicts future energy demand & supply for a selected Indian state at a specific future time."""
    try:
        data = request.json
        state = data.get("state")

        if not state or state not in available_states:
            return jsonify({"error": "Invalid state selected"}), 400

        # 🔹 Define the forecast time
        forecast_time = datetime.now() + timedelta(hours=24)  # Predicting for the next 24 hours
        forecast_time_str = forecast_time.strftime("%Y-%m-%d %H:%M:%S")

        # 🔹 Simulating demand and supply predictions (Replace with actual ML model output)
        predicted_demand = np.random.randint(500, 5000)  # Example: Demand in MW
        predicted_supply = np.random.randint(500, 5000)  # Example: Supply in MW
        renewable_contribution = np.random.randint(20, 70)  # Renewable percentage
        peak_time = f"{np.random.randint(15, 21)}:{np.random.randint(0, 59):02d}"

        # 🔹 Determine the power status
        if predicted_supply > predicted_demand:
            status = "Surplus"
        elif predicted_supply < predicted_demand:
            status = "Deficit"
        else:
            status = "Balanced"

        # 🔹 Suggested measures to clear deficit
        measures = []
        if status == "Deficit":
            measures = [
                "Increase renewable energy utilization.",
                "Optimize power distribution efficiency.",
                "Import power from surplus states.",
                "Encourage demand-side energy management.",
                "Enhance grid storage capacity."
            ]

        forecast_data = {
            "forecast_time": forecast_time_str,
            "demand": predicted_demand,
            "supply": predicted_supply,
            "renewable": renewable_contribution,
            "peak_time": peak_time,
            "status": status,
            "measures": measures
        }

        return jsonify(forecast_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
