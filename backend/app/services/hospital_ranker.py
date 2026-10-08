"""
CareNavigator - Hospital ranking with OSRM

Pipeline:
  1. Haversine prefilter  -> cheap, keeps ~30 nearby candidates (specialty-aware)
  2. OSRM Table API       -> ONE call gives real road distance + drive time to all candidates
  3. Weighted scoring     -> time, specialty match, PMJAY, availability, rating
  4. OSRM Route API       -> geometry for the top hospital only (for the map)

Expected hospitals DataFrame columns:
  name, lat, lon, specialties (pipe separated, e.g. "cardiology|general medicine"),
  pmjay (bool), emergency (bool)
Optional columns: beds_available (int), rating (0-5)
"""
import math
import requests
import numpy as np
import pandas as pd

# Self-hosted OSRM (see setup commands). For quick tests only:
# "https://router.project-osrm.org"  (demo server, rate limited, not for production)
import os
from dotenv import load_dotenv
load_dotenv()
OSRM_URL = os.getenv("OSRM_URL", "http://localhost:5000")

# Weights per severity. Each score component is normalised to 0..1 (higher = better).
WEIGHTS = {
    "low":      dict(time=0.25, specialty=0.30, pmjay=0.20, beds=0.10, rating=0.15),
    "moderate": dict(time=0.30, specialty=0.30, pmjay=0.15, beds=0.15, rating=0.10),
    "high":     dict(time=0.40, specialty=0.30, pmjay=0.10, beds=0.15, rating=0.05),
    "critical": dict(time=0.65, specialty=0.20, pmjay=0.05, beds=0.10, rating=0.00),
}

# Map predicted disease -> specialty (extend this with your own mapping)
DISEASE_TO_SPECIALTY = {
    "heart attack": "cardiology",
    "stroke": "neurology",
    "pneumonia": "pulmonology",
    "dengue": "general medicine",
    "fracture": "orthopedics",
}


def haversine_km(latitude1, longitude1, latitude2, longitude2):
    """Vectorised straight-line distance in km."""
    r = 6371.0
    p1, p2 = np.radians(latitude1), np.radians(latitude2)
    dphi = p2 - p1
    dlmb = np.radians(longitude2) - np.radians(longitude1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def osrm_table(user_latitude, user_longitude, cand: pd.DataFrame):
    """One request: durations (s) and distances (m) from user to every candidate."""
    coords = [f"{user_longitude},{user_latitude}"] + [f"{lo},{la}" for la, lo in zip(cand.latitude, cand.longitude )]
    dest_idx = ";".join(str(i) for i in range(1, len(coords)))
    url = f"{OSRM_URL}/table/v1/driving/{';'.join(coords)}"
    r = requests.get(
        url,
        params={"sources": 0, "destinations": dest_idx, "annotations": "duration,distance"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")
    return np.array(data["durations"][0], dtype=float), np.array(data["distances"][0], dtype=float)


def osrm_route(user_latitude, user_longitude, hospital_latitude, hospital_longitude):
    """Full route geometry (GeoJSON) + turn-by-turn steps for the chosen hospital."""
    url = f"{OSRM_URL}/route/v1/driving/{user_longitude},{user_latitude};{hospital_longitude},{hospital_latitude}"
    r = requests.get(url, params={"overview": "full", "geometries": "geojson", "steps": "true"}, timeout=10)
    r.raise_for_status()
    route = r.json()["routes"][0]
    return {"duration_min": route["duration"] / 60, "distance_km": route["distance"] / 1000,
            "geometry": route["geometry"], "steps": route["legs"][0]["steps"]}


def _minmax_inverse(x):
    """Lower value -> higher score."""
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return np.ones_like(x) if rng == 0 else 1 - (x - x.min()) / rng


def rank_hospitals(hospitals: pd.DataFrame, user_latitude, user_longitude, disease, severity="moderate",
                   top_n=5, prefilter_n=30, max_radius_km=150, pmjay_only=False):
    severity = severity.lower()
    w = WEIGHTS[severity]
    specialty = DISEASE_TO_SPECIALTY.get(disease.lower(), "general medicine")

    df = hospitals.copy()
    if pmjay_only:
        df = df[df.pmjay]
    if severity == "critical":
        df = df[df.emergency]          # must have emergency/casualty

    # ---- Stage 1: haversine prefilter ----
    df["straight_km"] = haversine_km(user_latitude, user_longitude, df.latitude.values, df.longitude.values)
    df = df[df.straight_km <= max_radius_km]
    df["specialty_match"] = df.specialties.str.lower().str.contains(specialty, regex=False).astype(float)

    # Keep specialty hospitals preferentially, but never end up with an empty list
    df = df.sort_values(["specialty_match", "straight_km"], ascending=[False, True])
    spec = df[df.specialty_match == 1].nsmallest(prefilter_n, "straight_km")
    rest = df[df.specialty_match == 0].nsmallest(max(prefilter_n - len(spec), 10), "straight_km")
    cand = pd.concat([spec, rest]).reset_index(drop=True)
    if cand.empty:
        return cand

    # ---- Stage 2: real road time/distance ----
    dur, dist = osrm_table(user_latitude, user_longitude, cand)
    cand["drive_min"] = dur / 60
    cand["road_km"] = dist / 1000
    cand = cand.dropna(subset=["drive_min"])      # unreachable points come back as null

    # ---- Stage 3: scoring ----
    cand["s_time"] = _minmax_inverse(cand.drive_min)
    cand["s_specialty"] = cand.specialty_match
    cand["s_pmjay"] = cand.pmjay.astype(float)
    cand["s_beds"] = (np.clip(cand.get("beds_available", 0), 0, 20) / 20) if "beds_available" in cand else 0.5
    cand["s_rating"] = (cand["rating"] / 5) if "rating" in cand else 0.5

    cand["score"] = (w["time"] * cand.s_time + w["specialty"] * cand.s_specialty +
                     w["pmjay"] * cand.s_pmjay + w["beds"] * cand.s_beds +
                     w["rating"] * cand.s_rating)

    # Critical: time dominates; tie-break by fastest arrival
    sort_cols = ["score", "drive_min"]
    cand = cand.sort_values(sort_cols, ascending=[False, True]).head(top_n).reset_index(drop=True)
    return cand[["name", "latitude", "longitude", "drive_min", "road_km", "specialty_match", "pmjay",
                 "emergency", "score"]]


if __name__ == "__main__":
    hospitals = pd.read_csv("C:\\CareNavigator\\ml\\data\\hospitals_tg_clean.csv")
    hospitals["pmjay"] = hospitals["pmjay"].astype(bool)
    hospitals["emergency"] = hospitals["emergency"].astype(bool)

    user = (17.4399, 78.4983)  # patient GPS (lat, lon)
    result = rank_hospitals(hospitals, *user, disease="heart attack", severity="critical")
    print(result.round(2).to_string(index=False))

    best = result.iloc[0]
    route = osrm_route(*user, best.latitude, best.longitude)
    print(f"\nBest: {best['name']} - {route['duration_min']:.1f} min, {route['distance_km']:.1f} km")