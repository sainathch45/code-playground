import folium
import json

def create_pollution_map(city_aqi_dict, cities_path='cities.json'):
    with open(cities_path) as f:
        cities = json.load(f)

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

    for city, coords in cities.items():
        aqi = city_aqi_dict.get(city, 0)
        color = 'red' if aqi > 150 else 'orange' if aqi > 100 else 'green'
        
        folium.CircleMarker(
            location=coords,
            radius=10,
            popup=f"{city}: PM2.5 = {aqi}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    m.save('map.html')
