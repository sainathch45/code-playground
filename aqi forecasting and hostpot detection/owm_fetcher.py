import requests
import pandas as pd

def get_coordinates(city_name):
    """Get latitude and longitude of the city from OWM Geocoding API"""
    url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": f"{city_name},IN",
        "limit": 1,
        "appid": "3e7a81887083bf159455a13ea8b2d263"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if data:
        return data[0]['lat'], data[0]['lon']
    return None, None

def get_current_aqi(lat, lon, api_key):
    """Fetch current AQI and PM2.5 for given coordinates"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(url, params=params)
    data = r.json()
    aqi = data['list'][0]['main']['aqi']
    pm2_5 = data['list'][0]['components']['pm2_5']
    return aqi, pm2_5

def get_forecast_aqi(lat, lon, api_key):
    """Fetch hourly PM2.5 forecast for given coordinates"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(url, params=params)
    data = r.json()

    records = []
    for entry in data['list']:
        dt = pd.to_datetime(entry['dt'], unit='s')
        pm2_5 = entry['components']['pm2_5']
        records.append({"datetime": dt, "pm2_5": pm2_5})

    df = pd.DataFrame(records)
    return df.set_index("datetime")
