"""
Fetches live real-world risk signals (weather + news) and writes them into
the risk_events table. Run on a schedule (cron / Airflow / simple loop)
to keep SCOUT's risk data current.

Requires free API keys, set as environment variables:
  OPENWEATHER_API_KEY   -> https://openweathermap.org/api
  NEWSAPI_KEY           -> https://newsapi.org

Usage: python data/scripts/fetch_live_signals.py
"""
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from db_connect import get_pymysql_connection

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
from db_connect import get_pymysql_connection

# Regions SCOUT is monitoring — extend based on your suppliers/shipments data
MONITORED_REGIONS = [
    {"name": "Mumbai", "lat": 19.076, "lon": 72.877},
    {"name": "Shanghai", "lat": 31.230, "lon": 121.474},
    {"name": "Rotterdam", "lat": 51.924, "lon": 4.478},
]


def fetch_weather_events():
    events = []
    for region in MONITORED_REGIONS:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={region['lat']}&lon={region['lon']}&appid={OPENWEATHER_KEY}&units=metric"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            weather_main = data["weather"][0]["main"]
            severity = "high" if weather_main in ("Thunderstorm", "Tornado", "Hurricane") else \
                       "medium" if weather_main in ("Rain", "Snow") else "low"
            events.append({
                "event_type": "weather",
                "source": "OpenWeatherMap",
                "description": f"{weather_main} in {region['name']}: {data['weather'][0]['description']}",
                "severity": severity,
                "region": region["name"],
                "event_date": datetime.utcnow(),
            })
    return events


def fetch_news_events(query="port strike OR supply chain disruption OR shipping delay"):
    events = []
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWSAPI_KEY}"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        for article in resp.json().get("articles", []):
            events.append({
                "event_type": "news",
                "source": article.get("source", {}).get("name", "NewsAPI"),
                "description": article.get("title", "")[:500],
                "severity": "medium",   # refine later with NLP severity classifier
                "region": "global",
                "event_date": datetime.utcnow(),
            })
    return events


def insert_events(events):
    if not events:
        print("No events to insert.")
        return
    conn = get_pymysql_connection()
    with conn.cursor() as cur:
        sql = """
            INSERT INTO risk_events (event_type, source, description, severity, region, event_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        for e in events:
            cur.execute(sql, (
                e["event_type"], e["source"], e["description"],
                e["severity"], e["region"], e["event_date"],
            ))
    conn.close()
    print(f"Inserted {len(events)} risk events.")


if __name__ == "__main__":
    all_events = fetch_weather_events() + fetch_news_events()
    insert_events(all_events)
