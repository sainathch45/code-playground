import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

@st.cache_data(show_spinner=False)
def forecast_aqi_lstm(df, n_input=24, n_predict=24):
    df = df[['pm2_5']].copy()

    # Drop NaNs
    df.dropna(inplace=True)
    if df.shape[0] < n_input:
        raise ValueError("Not enough data for forecasting. Need at least {} data points.".format(n_input))

    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Prepare sequences
    X, y = [], []
    for i in range(len(df_scaled) - n_input):
        X.append(df_scaled[i:i+n_input])
        y.append(df_scaled[i+n_input])
    X, y = np.array(X), np.array(y)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Define LSTM Model
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0, batch_size=8)

    # Forecast future
    last_input = df_scaled[-n_input:]
    preds = []
    input_seq = last_input.copy()

    for _ in range(n_predict):
        pred = model.predict(input_seq.reshape(1, n_input, 1), verbose=0)[0][0]
        preds.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    forecast_index = pd.date_range(start=df.index[-1], periods=n_predict+1, freq='H')[1:]
    forecast_df = pd.DataFrame({'datetime': forecast_index, 'predicted_pm2_5': preds})
    forecast_df.set_index('datetime', inplace=True)

    return forecast_df