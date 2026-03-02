

from __future__ import annotations

import re
import time
import threading
from datetime import datetime
from typing import Optional, Tuple

import requests


_WMO_CODES: dict[int, str] = {
    0:  "clear sky",
    1:  "mainly clear",  2: "partly cloudy",  3: "overcast",
    45: "foggy",         48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain",    63: "moderate rain",    65: "heavy rain",
    71: "light snow",    73: "moderate snow",    75: "heavy snow",
    77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm",  96: "thunderstorm with hail", 99: "severe thunderstorm",
}

# ---------------------------------------------------------------------------
# Regex patterns that signal a weather/today query
# ---------------------------------------------------------------------------
_WEATHER_PATTERNS = re.compile(
    r"\b("
    r"weather|temperature|temp\b|forecast|rain(ing)?|snow(ing)?|sunny|cloudy|humid"
    r"|hot\b|cold\b|warm\b|cool\b|wind(y)?\b|storm"
    r"|what('s| is) (it like|the weather|outside)|how('s| is) (the weather|outside)"
    r"|do i need (an? )?(umbrella|jacket|coat)"
    r")\b",
    re.IGNORECASE,
)

_TODAY_PATTERNS = re.compile(
    r"\b(today|tonight|this (morning|afternoon|evening)|right now|currently|outside)\b",
    re.IGNORECASE,
)


class WeatherClient:
   

    _GEO_URL     = "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon"
    _WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        location: Optional[str] = None,      # e.g. "London" — overrides auto-detect
        lat: Optional[float]    = None,       # explicit lat/lon skips ip-api lookup
        lon: Optional[float]    = None,
        units: str              = "celsius",  # "celsius" or "fahrenheit"
        cache_ttl: int          = 600,        # seconds to cache weather data
        timeout: float          = 4.0,        # per-request network timeout
    ) -> None:
        self._explicit_location = location
        self._explicit_lat = lat
        self._explicit_lon = lon
        self._units = units.lower()
        self._cache_ttl = int(cache_ttl)
        self._timeout   = float(timeout)

        # Resolved geo info
        self._city:    Optional[str]   = location
        self._lat:     Optional[float] = lat
        self._lon:     Optional[float] = lon
        self._geo_ok   = (lat is not None and lon is not None)
        self._geo_lock = threading.Lock()

        # Cached weather payload
        self._cache:      Optional[dict]  = None
        self._cache_time: float           = 0.0
        self._cache_lock  = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_weather_query(self, text: str) -> bool:
        """Return True if the transcribed text is asking about weather / today's conditions."""
        text = (text or "").strip()
        if not text:
            return False
        has_weather = bool(_WEATHER_PATTERNS.search(text))
        has_today   = bool(_TODAY_PATTERNS.search(text))
        return has_weather or has_today

    def get_summary(self) -> str:
        """
        Return a compact, voice-friendly weather summary string, e.g.:
          "Partly cloudy, 24 °C, wind 14 km/h. It's Tuesday, 3 PM."
        Returns an error string if the network is unavailable.
        """
        try:
            data = self._fetch_weather()
        except Exception as exc:
            return f"Sorry, I couldn't fetch the weather right now ({exc})."

        if data is None:
            return "Sorry, weather data is not available."

        current  = data.get("current_weather", {})
        temp     = current.get("temperature")
        wind     = current.get("windspeed")
        code     = int(current.get("weathercode", -1))
        condition = _WMO_CODES.get(code, "unknown conditions")

        unit_label = "°C" if self._units == "celsius" else "°F"
        wind_unit  = "km/h"

        parts: list[str] = []
        if condition:
            parts.append(condition.capitalize())
        if temp is not None:
            parts.append(f"{temp:.0f} {unit_label}")
        if wind is not None:
            parts.append(f"wind {wind:.0f} {wind_unit}")

        location_str = self._city or "your location"
        now = datetime.now()
        time_str = now.strftime("It's %A, %-I %p").replace(" 0", " ")  # "It's Tuesday, 3 PM"

        weather_str = ", ".join(parts) if parts else "conditions unknown"
        return f"In {location_str}: {weather_str}. {time_str}."

    def get_prompt_context(self) -> str:
        
        try:
            data = self._fetch_weather()
        except Exception:
            data = None

        now_str = datetime.now().strftime("%A, %d %b %Y, %-I:%M %p")

        if data is None:
            return f"[CONTEXT: Today is {now_str}. Live weather unavailable.]"

        current  = data.get("current_weather", {})
        temp     = current.get("temperature")
        wind     = current.get("windspeed")
        code     = int(current.get("weathercode", -1))
        condition = _WMO_CODES.get(code, "unknown")

        unit_label = "°C" if self._units == "celsius" else "°F"

        parts: list[str] = []
        if condition:
            parts.append(condition)
        if temp is not None:
            parts.append(f"{temp:.0f} {unit_label}")
        if wind is not None:
            parts.append(f"wind {wind:.0f} km/h")

        location_str = self._city or "your location"
        weather_str  = ", ".join(parts)

        return (
            f"[LIVE WEATHER — In {location_str}: {weather_str}. "
            f"Today is {now_str}. "
            f"Use this data to answer the user's weather or time question accurately.]"
        )

    # ------------------------------------------------------------------
    # Internal: geo-location
    # ------------------------------------------------------------------

    def _ensure_geo(self) -> bool:
        """Resolve lat/lon exactly once.  Returns True if geo is available."""
        with self._geo_lock:
            if self._geo_ok:
                return True
            try:
                r = requests.get(self._GEO_URL, timeout=self._timeout)
                r.raise_for_status()
                j = r.json()
                if j.get("status") == "success":
                    self._lat  = float(j["lat"])
                    self._lon  = float(j["lon"])
                    self._city = f"{j.get('city', '')}, {j.get('regionName', '')}".strip(", ")
                    self._geo_ok = True
                    print(f"[Weather] Auto-detected location: {self._city} "
                          f"({self._lat:.2f}, {self._lon:.2f})")
                else:
                    print("[Weather] ip-api.com geo-detection failed; set lat/lon manually.")
                    self._geo_ok = False
            except Exception as exc:
                print(f"[Weather] Geo-detection error: {exc}")
                self._geo_ok = False
        return self._geo_ok

    # ------------------------------------------------------------------
    # Internal: weather fetch + cache
    # ------------------------------------------------------------------

    def _fetch_weather(self) -> Optional[dict]:
        """Return cached or freshly fetched Open-Meteo payload."""
        with self._cache_lock:
            if self._cache is not None and (time.time() - self._cache_time) < self._cache_ttl:
                return self._cache

        if not self._ensure_geo():
            return None

        temp_unit = "celsius" if self._units == "celsius" else "fahrenheit"
        params = {
            "latitude":         self._lat,
            "longitude":        self._lon,
            "current_weather":  "true",
            "temperature_unit": temp_unit,
            "windspeed_unit":   "kmh",
            "timezone":         "auto",
        }
        r = requests.get(self._WEATHER_URL, params=params, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()

        with self._cache_lock:
            self._cache      = data
            self._cache_time = time.time()

        return data


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wc = WeatherClient()

    test_queries = [
        "How's the weather today?",
        "Do I need an umbrella?",
        "What time is it?",               # ← "time" alone won't trigger (by design)
        "Is it going to rain tonight?",
        "Tell me a joke",                 # ← should NOT trigger
        "How hot is it outside?",
    ]
    print("=== Query detection ===")
    for q in test_queries:
        print(f"  {'✓' if wc.is_weather_query(q) else '✗'}  {q}")

    print("\n=== Live summary ===")
    print(wc.get_summary())

    print("\n=== LLM prompt context ===")
    print(wc.get_prompt_context())