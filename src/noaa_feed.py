"""
Weather forecast fetcher — NOAA primary, Open-Meteo fallback.

Both are free with no API key required.
Converts forecasts into probability distributions over temperature buckets
that match Kalshi weather market ranges.

NOAA gridpoints: https://api.weather.gov/points/{lat},{lon}
Open-Meteo docs: https://open-meteo.com/en/docs
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 15
_SIGMA_F = 3.0  # NWS 1-2 day forecast std dev ≈ 3°F

# City metadata: (lat, lon, nws_office, nws_gridX, nws_gridY, tz)
# NWS gridpoints from: https://api.weather.gov/points/{lat},{lon}
CITIES: Dict[str, Dict] = {
    "new_york": {
        "lat": 40.7128, "lon": -74.0060,
        "nws_office": "OKX", "nws_gx": 33, "nws_gy": 37,
        "tz": "America/New_York",
    },
    "chicago": {
        "lat": 41.8781, "lon": -87.6298,
        "nws_office": "LOT", "nws_gx": 73, "nws_gy": 71,
        "tz": "America/Chicago",
    },
    "los_angeles": {
        "lat": 34.0522, "lon": -118.2437,
        "nws_office": "LOX", "nws_gx": 149, "nws_gy": 48,
        "tz": "America/Los_Angeles",
    },
    "miami": {
        "lat": 25.7617, "lon": -80.1918,
        "nws_office": "MFL", "nws_gx": 104, "nws_gy": 70,
        "tz": "America/New_York",
    },
    "boston": {
        "lat": 42.3601, "lon": -71.0589,
        "nws_office": "BOX", "nws_gx": 71, "nws_gy": 90,
        "tz": "America/New_York",
    },
}


@dataclass
class TempForecast:
    city: str
    forecast_time: str      # ISO-8601 start of forecast period
    high_f: float
    low_f: float
    source: str             # "noaa" or "open-meteo"
    fetched_at: float = field(default_factory=time.time)

    @property
    def mid_f(self) -> float:
        return (self.high_f + self.low_f) / 2.0

    def bucket_probs(self, bucket_size: int = 5) -> Dict[str, float]:
        """
        Probability distribution over temperature buckets of `bucket_size`°F.

        Model: forecast ~ N(mid_f, sigma=3°F). Integrates normal PDF over
        each bucket and normalises to sum to 1.0.

        Returns e.g. {"50-55": 0.72, "55-60": 0.21, ...}
        """
        mu = self.mid_f

        def _cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf((x - mu) / (_SIGMA_F * math.sqrt(2))))

        lo_bound = int((mu - 25) // bucket_size) * bucket_size
        hi_bound = int((mu + 25) // bucket_size) * bucket_size + bucket_size

        raw: Dict[str, float] = {}
        for lo in range(lo_bound, hi_bound, bucket_size):
            p = _cdf(lo + bucket_size) - _cdf(lo)
            if p > 0.001:
                raw[f"{lo}-{lo + bucket_size}"] = p

        total = sum(raw.values())
        return {k: round(v / total, 4) for k, v in raw.items()} if total else {}


class NOAAFeed:
    """
    Fetches temperature forecasts for configured cities.
    Tries NOAA first; falls back to Open-Meteo on failure.
    Results are cached for `cache_ttl` seconds.
    """

    def __init__(self, cities: List[str], cache_ttl: int = 1800):
        bad = [c for c in cities if c not in CITIES]
        if bad:
            raise ValueError(f"Unknown cities: {bad}. Add them to CITIES.")
        self._cities = cities
        self._ttl    = cache_ttl
        self._cache: Dict[str, Tuple[float, TempForecast]] = {}

    def fetch(self, city: str) -> Optional[TempForecast]:
        cached = self._cache.get(city)
        if cached and (time.time() - cached[0]) < self._ttl:
            return cached[1]
        fc = self._fetch_noaa(city) or self._fetch_open_meteo(city)
        if fc:
            self._cache[city] = (time.time(), fc)
        return fc

    def fetch_all(self) -> Dict[str, Optional[TempForecast]]:
        return {c: self.fetch(c) for c in self._cities}

    # ------------------------------------------------------------------
    # NOAA (primary)
    # ------------------------------------------------------------------

    def _fetch_noaa(self, city: str) -> Optional[TempForecast]:
        meta = CITIES[city]
        office, gx, gy = meta["nws_office"], meta["nws_gx"], meta["nws_gy"]
        url = f"https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast"
        headers = {
            "User-Agent": "WeatherBot/1.0 (weather arbitrage bot, contact@example.com)",
            "Accept": "application/geo+json",
        }
        try:
            r = requests.get(url, headers=headers, timeout=_TIMEOUT)
            r.raise_for_status()
            periods = r.json()["properties"]["periods"]
            daytime = [p for p in periods if p.get("isDaytime", True)]
            nightly = [p for p in periods if not p.get("isDaytime", True)]
            day     = daytime[0] if daytime else periods[0]
            night   = nightly[0] if nightly else None
            high_f  = float(day["temperature"])
            low_f   = float(night["temperature"]) if night else high_f - 10.0
            fc = TempForecast(
                city=city,
                forecast_time=day["startTime"],
                high_f=high_f,
                low_f=low_f,
                source="noaa",
            )
            log.info("[NOAA] %-15s high=%.1f°F  low=%.1f°F", city, high_f, low_f)
            return fc
        except Exception as exc:
            log.debug("NOAA failed [%s]: %s — trying Open-Meteo", city, exc)
            return None

    # ------------------------------------------------------------------
    # Open-Meteo fallback (free, no key)
    # ------------------------------------------------------------------

    def _fetch_open_meteo(self, city: str) -> Optional[TempForecast]:
        meta = CITIES[city]
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":         meta["lat"],
                    "longitude":        meta["lon"],
                    "daily":            "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone":         meta["tz"],
                    "forecast_days":    2,
                },
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data   = r.json()["daily"]
            high_f = float(data["temperature_2m_max"][0])
            low_f  = float(data["temperature_2m_min"][0])
            fc = TempForecast(
                city=city,
                forecast_time=data["time"][0],
                high_f=high_f,
                low_f=low_f,
                source="open-meteo",
            )
            log.info("[Open-Meteo] %-15s high=%.1f°F  low=%.1f°F", city, high_f, low_f)
            return fc
        except Exception as exc:
            log.warning("Open-Meteo failed [%s]: %s", city, exc)
            return None
