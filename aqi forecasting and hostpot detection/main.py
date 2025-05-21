import sys
print("Interpreter path:", sys.executable)

import streamlit as st
import pandas as pd
from owm_fetcher import get_coordinates, get_current_aqi, get_forecast_aqi
from model import forecast_aqi_lstm

import cities
import folium

OWM_KEY = "3e7a81887083bf159455a13ea8b2d263"

st.set_page_config(layout="wide")
st.title("🌫️ AirGuard India – Real-Time AQI Monitor (via OpenWeatherMap)")

# --- City Selection ---
city_names = list(cities.city_coords.keys())
selected_city = st.selectbox("Select a city", city_names)

lat, lon = get_coordinates(selected_city)
if not lat:
    st.error("Could not find city coordinates.")
    st.stop()

# --- Current AQI & PM2.5 ---
aqi_rating = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}
aqi_level, pm25 = get_current_aqi(lat, lon, OWM_KEY)
st.metric("🧪 Current PM2.5", f"{pm25} µg/m³")
st.metric("🌡️ AQI Level", f"{aqi_level} - {aqi_rating[aqi_level]}")

# --- Forecast Chart ---
st.subheader("📈 PM2.5 Forecast (Next 5 Days)")
forecast_df = get_forecast_aqi(lat, lon, OWM_KEY)
lstm_forecast_df = forecast_aqi_lstm(forecast_df)
st.line_chart(lstm_forecast_df['predicted_pm2_5'])


# --- Pollution Hotspot Map ---
st.subheader("🗺️ Pollution Hotspots Across India (Live)")

from streamlit_folium import st_folium
m = folium.Map(location=[22.97, 78.65], zoom_start=5)

for city, coord in cities.city_coords.items():
    lat, lon = coord
    try:
        _, pm25 = get_current_aqi(lat, lon, OWM_KEY)
        color = 'green' if pm25 <= 30 else 'orange' if pm25 <= 90 else 'red'
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"{city}: {pm25} µg/m³",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    except:
        pass

st_folium(m, width=900, height=600)
