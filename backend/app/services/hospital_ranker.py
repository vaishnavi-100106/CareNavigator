"""
CareNavigator - Hospital Ranking with OSRM

Pipeline:
1. Haversine prefilter -> cheap nearby candidates
2. OSRM Table API -> road distance + drive time
3. Weighted scoring -> time, specialty, PMJAY, availability, rating
4. OSRM Route API -> geometry for top hospital

Expected columns:
name / hospital_name
lat / latitude
lon / longitude
specialties
pmjay
emergency

Optional:
beds_available
rating
"""

import os
import math
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
OSRM_URL = os.getenv("OSRM_URL", "http://localhost:5000")

WEIGHTS = {
    "low": {"time": 0.25, "specialty": 0.30, "pmjay": 0.20, "beds": 0.10, "rating": 0.15},
    "moderate": {"time": 0.30, "specialty": 0.30, "pmjay": 0.15, "beds": 0.15, "rating": 0.10},
    "high": {"time": 0.40, "specialty": 0.30, "pmjay": 0.10, "beds": 0.15, "rating": 0.05},
    "critical": {"time": 0.65, "specialty": 0.20, "pmjay": 0.05, "beds": 0.10, "rating": 0.00},
}

DISEASE_TO_SPECIALTY = {
    "heart attack": "cardiology",
    "stroke": "neurology",
    "pneumonia": "pulmonology",
    "dengue": "general medicine",
    "fracture": "orthopedics",
}


def normalize_hospital_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize hospital CSV columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}
    if "hospital_name" in df.columns and "name" not in df.columns:
        rename_map["hospital_name"] = "name"
    elif "Hospital_Name" in df.columns and "name" not in df.columns:
        rename_map["Hospital_Name"] = "name"
    elif "hospital" in df.columns and "name" not in df.columns:
        rename_map["hospital"] = "name"

    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    elif "Latitude" in df.columns and "latitude" not in df.columns:
        rename_map["Latitude"] = "latitude"

    if "lon" in df.columns and "longitude" not in df.columns:
        rename_map["lon"] = "longitude"
    elif "lng" in df.columns and "longitude" not in df.columns:
        rename_map["lng"] = "longitude"
    elif "Longitude" in df.columns and "longitude" not in df.columns:
        rename_map["Longitude"] = "longitude"

    df = df.rename(columns=rename_map)

    required = ["name", "latitude", "longitude", "specialties", "pmjay", "emergency"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing required hospital columns: {missing}\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    df["name"] = df["name"].fillna("Unknown Hospital").astype(str)
    df["specialties"] = df["specialties"].fillna("").astype(str)

    bool_map = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False
    }

    df["pmjay"] = (
        df["pmjay"].astype(str).str.lower().map(bool_map).fillna(False)
    )
    df["emergency"] = (
        df["emergency"].astype(str).str.lower().map(bool_map).fillna(False)
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def haversine_km(latitude1, longitude1, latitude2, longitude2):
    """Vectorized straight-line distance in km."""
    r = 6371.0
    p1, p2 = np.radians(latitude1), np.radians(latitude2)
    dphi = p2 - p1
    dlmb = np.radians(longitude2) - np.radians(longitude1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


def osrm_table(user_latitude, user_longitude, cand: pd.DataFrame):
    """Get road distance and duration from user to candidates."""
    coords = [f"{user_longitude},{user_latitude}"]
    coords += [
        f"{lon},{lat}"
        for lat, lon in zip(cand["latitude"], cand["longitude"])
    ]

    url = f"{OSRM_URL}/table/v1/driving/{';'.join(coords)}"
    dest_idx = ";".join(str(i) for i in range(1, len(coords)))

    response = requests.get(
        url,
        params={
            "sources": 0,
            "destinations": dest_idx,
            "annotations": "duration,distance",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    durations = np.array(data["durations"][0], dtype=float)
    distances = np.array(data["distances"][0], dtype=float)
    return durations, distances


def osrm_route(
    user_latitude,
    user_longitude,
    hospital_latitude,
    hospital_longitude,
):
    """Get route geometry and turn-by-turn steps."""
    url = (
        f"{OSRM_URL}/route/v1/driving/"
        f"{user_longitude},{user_latitude};"
        f"{hospital_longitude},{hospital_latitude}"
    )

    response = requests.get(
        url,
        params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM route error: {data}")

    route = data["routes"][0]
    return {
        "duration_min": route["duration"] / 60,
        "distance_km": route["distance"] / 1000,
        "geometry": route["geometry"],
        "steps": route["legs"][0]["steps"],
    }


def _minmax_inverse(x):
    """Lower value -> higher score."""
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()

    if rng == 0:
        return np.ones_like(x)

    return 1 - ((x - x.min()) / rng)


def rank_hospitals(
    hospitals: pd.DataFrame,
    user_latitude,
    user_longitude,
    disease,
    severity="moderate",
    top_n=5,
    prefilter_n=30,
    max_radius_km=150,
    pmjay_only=False,
):
    severity = severity.lower()

    if severity not in WEIGHTS:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Use one of: {list(WEIGHTS.keys())}"
        )

    w = WEIGHTS[severity]
    disease = disease.lower()
    specialty = DISEASE_TO_SPECIALTY.get(disease, "general medicine")

    df = normalize_hospital_columns(hospitals)

    if pmjay_only:
        df = df[df["pmjay"]]

    if severity == "critical":
        df = df[df["emergency"]]

    if df.empty:
        print("No hospitals available after PMJAY/emergency filtering.")
        return pd.DataFrame()

    # Stage 1: Haversine prefilter
    df["straight_km"] = haversine_km(
        user_latitude,
        user_longitude,
        df["latitude"].values,
        df["longitude"].values,
    )

    df = df[df["straight_km"] <= max_radius_km].copy()

    if df.empty:
        return pd.DataFrame()

    # Specialty matching
    df["specialty_match"] = (
        df["specialties"]
        .str.lower()
        .str.contains(specialty, regex=False, na=False)
        .astype(float)
    )

    # Prefer specialty hospitals
    df = df.sort_values(
        ["specialty_match", "straight_km"],
        ascending=[False, True],
    )

    spec = df[df["specialty_match"] == 1].nsmallest(
        prefilter_n, "straight_km"
    )

    remaining_n = max(prefilter_n - len(spec), 10)

    rest = df[df["specialty_match"] == 0].nsmallest(
        remaining_n, "straight_km"
    )

    cand = (
        pd.concat([spec, rest])
        .drop_duplicates(subset=["name"])
        .reset_index(drop=True)
    )

    if cand.empty:
        return pd.DataFrame()

    print(f"\nOSRM routing {len(cand)} candidate hospitals...")

    # Stage 2: OSRM road distance/time
    dur, dist = osrm_table(
        user_latitude,
        user_longitude,
        cand,
    )

    cand["drive_min"] = dur / 60
    cand["road_km"] = dist / 1000

    cand = cand.dropna(subset=["drive_min"]).copy()

    if cand.empty:
        return pd.DataFrame()

    # Stage 3: scoring
    cand["s_time"] = _minmax_inverse(cand["drive_min"])
    cand["s_specialty"] = cand["specialty_match"]
    cand["s_pmjay"] = cand["pmjay"].astype(float)

    if "beds_available" in cand.columns:
        beds = pd.to_numeric(
            cand["beds_available"], errors="coerce"
        ).fillna(0)
        cand["s_beds"] = np.clip(beds, 0, 20) / 20
    else:
        cand["s_beds"] = 0.5

    if "rating" in cand.columns:
        rating = pd.to_numeric(
            cand["rating"], errors="coerce"
        ).fillna(0)
        cand["s_rating"] = rating / 5
    else:
        cand["s_rating"] = 0.5

    # Final weighted score
    cand["score"] = (
        w["time"] * cand["s_time"]
        + w["specialty"] * cand["s_specialty"]
        + w["pmjay"] * cand["s_pmjay"]
        + w["beds"] * cand["s_beds"]
        + w["rating"] * cand["s_rating"]
    )

    # Sort and select top hospitals
    cand = (
        cand.sort_values(
            ["score", "drive_min"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    output_columns = [
        "name",
        "latitude",
        "longitude",
        "drive_min",
        "road_km",
        "specialty_match",
        "pmjay",
        "emergency",
        "score",
    ]

    return cand[[col for col in output_columns if col in cand.columns]]


if __name__ == "__main__":
    hospital_file = r"C:\CareNavigator\ml\data\hospitals_tg_clean.csv"

    print("\nLoading hospitals...")
    hospitals = pd.read_csv(hospital_file)

    print("Original CSV columns:")
    print(hospitals.columns.tolist())

    hospitals = normalize_hospital_columns(hospitals)

    print("\nNormalized columns:")
    print(hospitals.columns.tolist())

    # Patient location
    user = (17.4399, 78.4983)

    # Rank hospitals
    result = rank_hospitals(
        hospitals,
        *user,
        disease="heart attack",
        severity="critical",
    )

    if result.empty:
        print("\nNo suitable hospitals found.")
        exit()

    print("\n==============================")
    print("TOP HOSPITALS")
    print("==============================")
    print(result.round(2).to_string(index=False))

    # Best hospital
    best = result.iloc[0]

    # Get actual OSRM route
    route = osrm_route(
        *user,
        best["latitude"],
        best["longitude"],
    )

    print("\n==============================")
    print("BEST HOSPITAL")
    print("==============================")
    print(f"Hospital      : {best['name']}")
    print(f"Road distance : {route['distance_km']:.2f} km")
    print(f"Drive time    : {route['duration_min']:.1f} min")
    print(f"Score         : {best['score']:.3f}")