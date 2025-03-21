import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("dataset/long_data_.csv")
df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True)

df.set_index("Dates", inplace=True)

scaler = MinMaxScaler()
df["Usage"] = scaler.fit_transform(df[["Usage"]])

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(df.index.factorize()[0].reshape(-1, 1), df["Usage"])

# Train LSTM model
X = []
y = []
sequence_length = 7

for i in range(len(df) - sequence_length):
    X.append(df["Usage"].iloc[i : i + sequence_length].values)
    y.append(df["Usage"].iloc[i + sequence_length])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X, y, epochs=10, batch_size=16)

# Save models
joblib.dump(xgb_model, "models/xgboost_model.pkl")
lstm_model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
