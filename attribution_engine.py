"""
attribution_engine.py
─────────────────────────────────────────────────────────────────────────────
Geospatial Pollution Source Attribution Engine for AirTwin X.

Estimates the percentage contribution of five pollution source categories
at any (lat, lon) point using transparent, auditable scoring sub-models.
All scores sum to 100 %. A confidence score (0–100) reflects how many live
data sources were available vs. how many fell back to heuristics.

Source categories
─────────────────
  1. Traffic             – road density + road type weights + time-of-day
  2. Industrial          – proximity to industrial OSM land-use zones
  3. Construction        – proximity to active construction OSM tags
  4. Biomass Burning     – NASA FIRMS fire hotspot count within radius
  5. Weather Amplification – wind stagnation + temperature inversion proxy
                            + humidity (from Open-Meteo telemetry)

Design principles
─────────────────
• No random weights.  Every score is computed from a named formula.
• Graceful degradation: if an external API is unavailable the engine
  substitutes a conservative heuristic and reduces the confidence score.
• Single responsibility: this module has no Streamlit imports and no
  side-effects.  It is pure computation.
• All public methods are documented with their inputs, outputs, and the
  reasoning behind every coefficient.
"""

from __future__ import annotations

import datetime
import math
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceScores:
    """
    Raw (un-normalised) scores for each pollution source at a given point.
    All values are non-negative; higher = larger contribution.
    """
    traffic: float = 0.0
    industrial: float = 0.0
    construction: float = 0.0
    biomass: float = 0.0
    weather: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Traffic": self.traffic,
            "Industrial": self.industrial,
            "Construction": self.construction,
            "Biomass Burning": self.biomass,
            "Weather Amplification": self.weather,
        }


@dataclass
class AttributionResult:
    """
    Fully resolved attribution for a location.

    percentages : dict mapping source name → contribution % (sums to 100)
    primary_source : name of the dominant source
    confidence : 0–100 integer reflecting data quality
    explanation : human-readable summary sentence
    sub_scores : raw SourceScores before normalisation (for debugging)
    data_sources_used : list of strings describing what data was live vs. heuristic
    """
    percentages: dict[str, float] = field(default_factory=dict)
    primary_source: str = ""
    confidence: int = 0
    explanation: str = ""
    sub_scores: SourceScores = field(default_factory=SourceScores)
    data_sources_used: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Road-type weights
# These mirror the ROAD_AQI_BASELINE in train_agent.py so the two modules
# remain conceptually consistent.
# ──────────────────────────────────────────────────────────────────────────────
ROAD_TRAFFIC_WEIGHT: dict[str, float] = {
    "motorway":    1.00,
    "trunk":       0.90,
    "primary":     0.75,
    "secondary":   0.55,
    "tertiary":    0.40,
    "residential": 0.20,
    "living_street": 0.10,
    "service":     0.15,
    "unclassified": 0.25,
}

# Rush-hour bands (hour-of-day → multiplier on traffic score)
RUSH_MULTIPLIER: dict[int, float] = {
    7: 1.30, 8: 1.60, 9: 1.55, 10: 1.25,
    17: 1.45, 18: 1.70, 19: 1.60, 20: 1.30,
}

# OSM land-use tags that indicate industrial pollution sources
INDUSTRIAL_LANDUSE = {"industrial", "port", "quarry", "landfill", "construction"}

# OSM building/amenity tags that indicate active construction dust
CONSTRUCTION_TAGS = {"construction", "building_construction"}

# Search radius (degrees ≈ km) for spatial queries around a point
SEARCH_RADIUS_DEG = 0.05   # ≈ 5.5 km

# NASA FIRMS CSV endpoint (no auth required for NRT data up to 7 days)
FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
    "/c3VwZXJzZWNyZXQ/VIIRS_SNPP_NRT/IND/1"
)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class SourceAttributionEngine:
    """
    Computes spatially-resolved pollution source attribution.

    Usage
    ─────
        engine = SourceAttributionEngine(graph=G)
        result = engine.attribute(lat=28.65, lon=77.22, telemetry=telemetry_dict)
    """

    def __init__(self, graph=None):
        """
        Parameters
        ──────────
        graph : networkx MultiDiGraph (OSMnx road graph)
            Already loaded and AQI-annotated by app.py.  The engine reads
            edge attributes ('highway', 'length', 'mock_aqi') — it never
            modifies them.
        """
        self._graph = graph
        # Cached OSM feature sets, fetched lazily per city
        self._industrial_zones: list[dict] = []
        self._construction_zones: list[dict] = []
        self._zones_loaded_for: Optional[tuple[float, float]] = None  # (lat, lon)
        # NASA FIRMS fire data, fetched once per session
        self._fire_hotspots: list[dict] = []
        self._firms_attempted: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def attribute(
        self,
        lat: float,
        lon: float,
        telemetry: Optional[dict],
        current_hour: Optional[int] = None,
        city_center: Optional[tuple[float, float]] = None,
    ) -> AttributionResult:
        """
        Main entry-point.  Computes full attribution for a map location.

        Parameters
        ──────────
        lat, lon       : decimal degrees of the point to attribute
        telemetry      : dict returned by fetch_live_telemetry() in app.py
                         Keys used: wind_speed, wind_dir, temp, active_fires
                         Pass None if unavailable.
        current_hour   : int 0-23.  Defaults to datetime.now().hour.
        city_center    : (lat, lon) of city centre for lazy zone loading.
                         Defaults to (lat, lon) itself.

        Returns
        ───────
        AttributionResult with all fields populated.
        """
        if current_hour is None:
            current_hour = datetime.datetime.now().hour
        if city_center is None:
            city_center = (lat, lon)

        data_sources: list[str] = []
        confidence_deductions: list[int] = []

        # ── 1. Load spatial feature layers (lazy, cached per city) ──────────
        self._load_spatial_zones(city_center, data_sources, confidence_deductions)

        # ── 2. Fetch NASA FIRMS fire data (lazy, cached per session) ────────
        self._load_fire_data(data_sources, confidence_deductions)

        # ── 3. Compute raw sub-scores ────────────────────────────────────────
        scores = SourceScores()

        scores.traffic = self._traffic_score(lat, lon, current_hour, data_sources)
        scores.industrial = self._industrial_score(lat, lon, data_sources)
        scores.construction = self._construction_score(lat, lon, data_sources)
        scores.biomass = self._biomass_score(lat, lon, telemetry, data_sources)
        scores.weather = self._weather_score(telemetry, data_sources, confidence_deductions)

        # ── 4. Normalise to percentages ──────────────────────────────────────
        percentages = self._normalise(scores)

        # ── 5. Compute confidence ────────────────────────────────────────────
        base_confidence = 85
        confidence = max(30, base_confidence - sum(confidence_deductions))

        # ── 6. Identify primary source ───────────────────────────────────────
        primary = max(percentages, key=percentages.get)  # type: ignore[arg-type]

        # ── 7. Build natural-language explanation ────────────────────────────
        explanation = self._build_explanation(
            primary, percentages, scores, lat, lon, telemetry, current_hour
        )

        return AttributionResult(
            percentages=percentages,
            primary_source=primary,
            confidence=confidence,
            explanation=explanation,
            sub_scores=scores,
            data_sources_used=data_sources,
        )

    def get_map_overlays(
        self,
        city_lat: float,
        city_lon: float,
    ) -> dict[str, list[dict]]:
        """
        Returns geo-features for map overlay rendering in app.py.

        Returns a dict:
          {
            "industrial"   : [{"lat":..., "lon":..., "name":...}, ...],
            "construction" : [...],
            "fire_hotspots": [...],
          }
        The caller (app.py) renders these as Folium markers.
        """
        data_sources: list[str] = []
        deductions: list[int] = []
        self._load_spatial_zones((city_lat, city_lon), data_sources, deductions)
        self._load_fire_data(data_sources, deductions)

        return {
            "industrial":    self._industrial_zones,
            "construction":  self._construction_zones,
            "fire_hotspots": self._fire_hotspots,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Spatial zone loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_spatial_zones(
        self,
        city_center: tuple[float, float],
        data_sources: list[str],
        confidence_deductions: list[int],
    ) -> None:
        """
        Loads industrial and construction zone centroids from Overpass API.
        Results are cached for the lifetime of the engine instance.
        Falls back to deriving approximate zones from the OSMnx graph when
        Overpass is unreachable.
        """
        if self._zones_loaded_for == city_center:
            return  # already loaded

        lat, lon = city_center
        self._industrial_zones = []
        self._construction_zones = []

        # Try Overpass for real OSM land-use polygons
        overpass_ok = False
        try:
            bbox = (lat - 0.15, lon - 0.15, lat + 0.15, lon + 0.15)
            query = (
                f"[out:json][timeout:20];"
                f"("
                f"  node[landuse~'industrial|port|quarry|landfill']"
                f"    ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});"
                f"  way[landuse~'industrial|port|quarry|landfill']"
                f"    ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});"
                f"  node[building=construction]"
                f"    ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});"
                f"  way[building=construction]"
                f"    ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});"
                f");"
                f"out center;"
            )
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=20,
            )
            if resp.status_code == 200:
                elements = resp.json().get("elements", [])
                for el in elements:
                    # Nodes have lat/lon directly; ways have 'center'
                    if el.get("type") == "node":
                        el_lat, el_lon = el.get("lat"), el.get("lon")
                    elif el.get("type") == "way" and "center" in el:
                        el_lat = el["center"]["lat"]
                        el_lon = el["center"]["lon"]
                    else:
                        continue

                    tags = el.get("tags", {})
                    landuse = tags.get("landuse", "")
                    building = tags.get("building", "")
                    name = tags.get("name", "")

                    if landuse in INDUSTRIAL_LANDUSE or building in {"industrial"}:
                        self._industrial_zones.append({
                            "lat": el_lat, "lon": el_lon,
                            "name": name or f"{landuse.title()} Zone",
                            "type": landuse or "industrial",
                        })
                    elif building == "construction" or landuse == "construction":
                        self._construction_zones.append({
                            "lat": el_lat, "lon": el_lon,
                            "name": name or "Active Construction Site",
                            "type": "construction",
                        })

                overpass_ok = len(elements) > 0
                if overpass_ok:
                    data_sources.append("Overpass/OSM land-use (live)")

        except Exception:
            pass

        if not overpass_ok:
            # Fallback: derive approximate industrial zones from graph edge
            # highway types — motorway/trunk corridors are reliable proxies
            # for industrial-adjacent infrastructure in Indian cities.
            self._industrial_zones = self._graph_derived_industrial_zones(lat, lon)
            self._construction_zones = self._graph_derived_construction_zones(lat, lon)
            data_sources.append("Road-graph proxy zones (OSM Overpass unavailable)")
            confidence_deductions.append(10)

        self._zones_loaded_for = city_center

    def _graph_derived_industrial_zones(
        self, city_lat: float, city_lon: float
    ) -> list[dict]:
        """
        When Overpass is unavailable, infer industrial zone proxies from the
        road graph: clusters of motorway/trunk edges far from the city centre
        tend to be near industrial corridors in South Asian cities.
        """
        if self._graph is None:
            return []

        zones: list[dict] = []
        seen_clusters: set[tuple[int, int]] = set()  # grid cell dedup

        for u, v, data in self._graph.edges(data=True):
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            if hw not in ("motorway", "trunk", "primary"):
                continue

            node_lat = self._graph.nodes[u].get("y", 0)
            node_lon = self._graph.nodes[u].get("x", 0)
            dist = _haversine_km(city_lat, city_lon, node_lat, node_lon)

            # Industrial proxies: major roads 2–8 km from city centre
            if 2.0 < dist < 8.0:
                cell = (int(node_lat * 20), int(node_lon * 20))  # 50m grid
                if cell not in seen_clusters:
                    seen_clusters.add(cell)
                    zones.append({
                        "lat": node_lat, "lon": node_lon,
                        "name": f"Industrial Corridor ({hw.title()})",
                        "type": "industrial_proxy",
                    })
            if len(zones) >= 12:
                break

        return zones

    def _graph_derived_construction_zones(
        self, city_lat: float, city_lon: float
    ) -> list[dict]:
        """
        Infer construction-activity proxies from secondary/tertiary edges in
        the 1–4 km band around the city centre (urban densification zone).
        """
        if self._graph is None:
            return []

        zones: list[dict] = []
        seen: set[tuple[int, int]] = set()

        for u, v, data in self._graph.edges(data=True):
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            if hw not in ("secondary", "tertiary"):
                continue

            node_lat = self._graph.nodes[u].get("y", 0)
            node_lon = self._graph.nodes[u].get("x", 0)
            dist = _haversine_km(city_lat, city_lon, node_lat, node_lon)

            if 1.0 < dist < 4.0:
                cell = (int(node_lat * 30), int(node_lon * 30))
                if cell not in seen:
                    seen.add(cell)
                    zones.append({
                        "lat": node_lat, "lon": node_lon,
                        "name": "Probable Construction Zone",
                        "type": "construction_proxy",
                    })
            if len(zones) >= 8:
                break

        return zones

    # ──────────────────────────────────────────────────────────────────────
    # NASA FIRMS fire data
    # ──────────────────────────────────────────────────────────────────────

    def _load_fire_data(
        self,
        data_sources: list[str],
        confidence_deductions: list[int],
    ) -> None:
        """
        Fetches NASA FIRMS VIIRS NRT fire hotspots for India (last 24h).
        Parses the CSV response into a list of {"lat", "lon", "frp"} dicts.
        Falls back to an empty list if the API is unreachable.
        """
        if self._firms_attempted:
            return
        self._firms_attempted = True

        try:
            resp = requests.get(FIRMS_URL, timeout=15)
            if resp.status_code == 200 and "latitude" in resp.text:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    lat_idx = headers.index("latitude")
                    lon_idx = headers.index("longitude")
                    frp_idx = headers.index("frp") if "frp" in headers else None

                    for line in lines[1:]:
                        parts = line.split(",")
                        try:
                            fire_lat = float(parts[lat_idx])
                            fire_lon = float(parts[lon_idx])
                            frp = float(parts[frp_idx]) if frp_idx is not None else 10.0
                            self._fire_hotspots.append({
                                "lat": fire_lat,
                                "lon": fire_lon,
                                "frp": frp,  # Fire Radiative Power (MW)
                            })
                        except (IndexError, ValueError):
                            continue

                    data_sources.append(f"NASA FIRMS VIIRS NRT ({len(self._fire_hotspots)} hotspots)")
                    return

        except Exception:
            pass

        # Fallback: use active_fires count from Open-Meteo heuristic in telemetry
        data_sources.append("Fire count: Open-Meteo thermal heuristic (FIRMS unavailable)")
        confidence_deductions.append(8)

    # ──────────────────────────────────────────────────────────────────────
    # Sub-score calculations
    # ──────────────────────────────────────────────────────────────────────

    def _traffic_score(
        self,
        lat: float,
        lon: float,
        hour: int,
        data_sources: list[str],
    ) -> float:
        """
        Traffic score = sum of (road_weight × length) for edges within
        SEARCH_RADIUS_DEG of the point, multiplied by rush-hour factor.

        Road weights are based on ROAD_TRAFFIC_WEIGHT — motorways score 1.0,
        residential roads score 0.2, reflecting vehicle throughput and
        emission density per unit length.

        Returns a dimensionless score in roughly [0, 400].
        """
        if self._graph is None:
            data_sources.append("Traffic: road-density heuristic (no graph)")
            return 60.0  # conservative urban baseline

        score = 0.0
        for u, v, edge_data in self._graph.edges(data=True):
            node_lat = self._graph.nodes[u].get("y", 0)
            node_lon = self._graph.nodes[u].get("x", 0)

            # Approximate degree-to-km: 1° ≈ 111 km
            dlat = abs(node_lat - lat)
            dlon = abs(node_lon - lon)
            if dlat > SEARCH_RADIUS_DEG or dlon > SEARCH_RADIUS_DEG:
                continue  # fast pre-filter before haversine

            dist_km = _haversine_km(lat, lon, node_lat, node_lon)
            if dist_km > SEARCH_RADIUS_DEG * 111:
                continue

            hw = edge_data.get("highway", "residential")
            if isinstance(hw, list):
                hw = hw[0]

            road_weight = ROAD_TRAFFIC_WEIGHT.get(hw, 0.20)
            length_m = float(edge_data.get("length", 50))

            # Inverse-distance weighting: closer roads count more
            dist_weight = 1.0 / (1.0 + dist_km)
            score += road_weight * (length_m / 100.0) * dist_weight

        rush_mult = RUSH_MULTIPLIER.get(hour, 1.0)
        final_score = score * rush_mult

        data_sources.append(
            f"Traffic: OSMnx road graph ({len(list(self._graph.edges()))} edges), "
            f"rush-hour factor {rush_mult:.2f}×"
        )
        return max(5.0, final_score)

    def _industrial_score(
        self,
        lat: float,
        lon: float,
        data_sources: list[str],
    ) -> float:
        """
        Industrial score = sum over all known industrial zone centroids of:
            intensity_weight / (1 + distance_km²)

        The inverse-square distance law approximates how an emission plume
        disperses: concentration ~ 1/r² under neutral stability conditions.

        Returns a dimensionless score in roughly [0, 200].
        """
        if not self._industrial_zones:
            return 10.0  # minimal background industrial

        score = 0.0
        for zone in self._industrial_zones:
            dist_km = _haversine_km(lat, lon, zone["lat"], zone["lon"])
            zone_type = zone.get("type", "industrial")
            # Heavy industry gets a higher intrinsic weight
            intensity = 2.5 if zone_type in ("industrial", "port") else 1.5
            score += intensity / (1.0 + dist_km ** 2)

        return score * 50.0  # scale to comparable range

    def _construction_score(
        self,
        lat: float,
        lon: float,
        data_sources: list[str],
    ) -> float:
        """
        Construction (dust) score = sum over construction zone centroids of:
            1.0 / (1 + distance_km)

        Linear distance decay (not squared) because construction dust
        disperses less efficiently than stack emissions — it is ground-level
        and turbulence-limited.

        Returns a dimensionless score in roughly [0, 60].
        """
        if not self._construction_zones:
            return 5.0

        score = 0.0
        for zone in self._construction_zones:
            dist_km = _haversine_km(lat, lon, zone["lat"], zone["lon"])
            score += 1.0 / (1.0 + dist_km)

        return score * 12.0

    def _biomass_score(
        self,
        lat: float,
        lon: float,
        telemetry: Optional[dict],
        data_sources: list[str],
    ) -> float:
        """
        Biomass burning score combines:
          (a) NASA FIRMS fire hotspot count and intensity within 200 km,
              weighted by Fire Radiative Power (FRP, in MW).
          (b) Wind transport factor: fires upwind contribute more than
              crosswind or downwind fires.
          (c) If FIRMS is unavailable: falls back to telemetry['active_fires'].

        Returns a dimensionless score in roughly [0, 300].
        """
        wind_dir = telemetry.get("wind_dir", 180) if telemetry else 180
        wind_speed = telemetry.get("wind_speed", 5) if telemetry else 5

        if self._fire_hotspots:
            score = 0.0
            for fire in self._fire_hotspots:
                dist_km = _haversine_km(lat, lon, fire["lat"], fire["lon"])
                if dist_km > 250:
                    continue  # fires beyond 250 km have negligible transport contribution

                frp = fire.get("frp", 10.0)
                # FRP-weighted inverse distance
                base = (frp / 10.0) / (1.0 + (dist_km / 50.0) ** 2)

                # Wind transport factor: bearing from fire to point
                bearing = _bearing(fire["lat"], fire["lon"], lat, lon)
                angular_diff = abs((bearing - wind_dir + 360) % 360)
                if angular_diff > 180:
                    angular_diff = 360 - angular_diff
                # cos(0) = 1 (directly downwind), cos(90°) = 0 (crosswind)
                wind_factor = max(0.1, math.cos(math.radians(angular_diff)))
                # Wind speed amplifies transport
                wind_amp = min(2.0, wind_speed / 10.0)

                score += base * wind_factor * wind_amp

            return score * 30.0

        # Fallback to telemetry heuristic
        active_fires = (telemetry or {}).get("active_fires", 0)
        return float(active_fires) * 12.0 + 3.0  # baseline non-zero

    def _weather_score(
        self,
        telemetry: Optional[dict],
        data_sources: list[str],
        confidence_deductions: list[int],
    ) -> float:
        """
        Weather amplification score quantifies how meteorological conditions
        trap or disperse pollutants:

          stagnation   = max(0, (threshold_wind - wind_speed) / threshold_wind)
                         High when wind is calm; dispersal is poor.
          temp_inversion_proxy = max(0, (temp - 25) / 20)
                         Warm days create surface inversions trapping PM2.5.
          humidity_factor = humidity / 100 * 0.4
                         High humidity grows aerosol particles (hygroscopic growth).

        All three components are normalised to [0, 1] and summed.
        Returns a dimensionless score in roughly [0, 120].
        """
        if telemetry is None:
            data_sources.append("Weather: no telemetry (default moderate conditions)")
            confidence_deductions.append(5)
            return 20.0

        wind_speed = telemetry.get("wind_speed", 10)
        temp = telemetry.get("temp", 25)
        # Humidity not in the current fetch_live_telemetry() call;
        # we derive it from temperature (warm & calm ≈ humid in Indian subcontinent context).
        humidity_proxy = max(0, min(100, 40 + (temp - 20) * 1.5))

        # Stagnation: below 8 km/h the boundary layer traps pollutants
        stagnation = max(0.0, (8.0 - wind_speed) / 8.0)
        # Temperature inversion proxy: above 30°C surface heating is significant
        temp_inv = max(0.0, (temp - 25.0) / 25.0)
        # Hygroscopic growth amplifier
        humidity_factor = humidity_proxy / 100.0 * 0.4

        raw = stagnation + temp_inv + humidity_factor
        data_sources.append(
            f"Weather: Open-Meteo (wind {wind_speed:.1f} km/h, "
            f"temp {temp:.1f}°C, humidity proxy {humidity_proxy:.0f}%)"
        )
        return raw * 40.0  # scale to comparable range

    # ──────────────────────────────────────────────────────────────────────
    # Normalisation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(scores: SourceScores) -> dict[str, float]:
        """
        Converts raw scores to percentages that sum to exactly 100 %.
        A minimum floor of 2 % per source prevents any source from vanishing
        entirely (even clean areas have some background from each type).
        """
        raw = scores.as_dict()
        FLOOR = 2.0
        floored = {k: max(FLOOR, v) for k, v in raw.items()}
        total = sum(floored.values())

        # Normalise to 100
        normalised = {k: (v / total) * 100.0 for k, v in floored.items()}

        # Re-normalise after rounding to guarantee exact sum of 100
        rounded = {k: round(v, 1) for k, v in normalised.items()}
        diff = 100.0 - sum(rounded.values())
        if abs(diff) > 0.001:
            # Add residual to the largest source
            largest = max(rounded, key=rounded.get)  # type: ignore[arg-type]
            rounded[largest] = round(rounded[largest] + diff, 1)

        return rounded

    # ──────────────────────────────────────────────────────────────────────
    # Explanation generation
    # ──────────────────────────────────────────────────────────────────────

    def _build_explanation(
        self,
        primary: str,
        percentages: dict[str, float],
        scores: SourceScores,
        lat: float,
        lon: float,
        telemetry: Optional[dict],
        hour: int,
    ) -> str:
        """
        Generates a single human-readable explanation sentence for the
        primary pollution driver, incorporating actual computed values.
        """
        pct = percentages[primary]
        wind = (telemetry or {}).get("wind_speed", 0)
        temp = (telemetry or {}).get("temp", 25)

        is_rush = hour in RUSH_MULTIPLIER and RUSH_MULTIPLIER[hour] > 1.2

        if primary == "Traffic":
            rush_phrase = "during peak rush hour" if is_rush else "due to high vehicle throughput"
            return (
                f"Traffic is responsible for {pct:.0f}% of local pollution {rush_phrase}. "
                f"High road density and major arterial corridors near this location generate "
                f"concentrated vehicular exhaust and tyre-wear particulates."
            )
        elif primary == "Industrial":
            return (
                f"Industrial emissions account for {pct:.0f}% of the pollution load at this point. "
                f"Proximity to industrial zones with stack emissions and heavy-vehicle activity "
                f"is the dominant driver."
            )
        elif primary == "Construction":
            return (
                f"Construction dust contributes {pct:.0f}% to local AQI. "
                f"Active building sites in the vicinity are generating coarse PM10 particulates "
                f"through earthmoving and material handling."
            )
        elif primary == "Biomass Burning":
            fire_count = len(self._fire_hotspots)
            fire_phrase = (
                f"{fire_count} active fire hotspots detected by NASA FIRMS"
                if fire_count > 0
                else "open burning activity"
            )
            return (
                f"Biomass burning drives {pct:.0f}% of the pollution signature. "
                f"Wind transport ({wind:.1f} km/h) is carrying smoke from "
                f"{fire_phrase} upwind of this location."
            )
        elif primary == "Weather Amplification":
            stag_phrase = (
                f"Low wind speeds ({wind:.1f} km/h) are suppressing dispersion"
                if wind < 8
                else f"Temperature ({temp:.0f}°C) is creating surface inversions"
            )
            return (
                f"Meteorological conditions are amplifying local pollution by {pct:.0f}%. "
                f"{stag_phrase}, trapping particulates in the boundary layer "
                f"and preventing dilution."
            )

        return f"{primary} is the primary driver at {pct:.0f}% of total pollution load at this location."


# ──────────────────────────────────────────────────────────────────────────────
# Spatial utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two (lat, lon) points in kilometres.
    Uses the Haversine formula — accurate to ~0.5% at urban scale.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Forward bearing (degrees, 0 = N, 90 = E) from point 1 to point 2.
    Used to compute the angular alignment between fire location and wind direction.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
