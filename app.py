import os
import re
import json
import requests
from urllib.parse import quote
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

MONTHS = [
    "July 2025",
    "August 2025",
    "September 2025",
    "October 2025",
    "November 2025",
    "December 2025",
]

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

if not os.getenv("MAPBOX_TOKEN"):
    raise RuntimeError("MAPBOX_TOKEN is not set in environment variables.")

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CSV_PATH = DATA_DIR / "MVA_Customers_Served_&_Wait_Time_by_Branch_20260303.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found at: {CSV_PATH}. Files in Data/: {[p.name for p in DATA_DIR.glob('*')]}"
    )

df = pd.read_csv(CSV_PATH)

# Coordinates for MVA branches (lat, lon)
BRANCH_COORDS = {
    "Annapolis": (38.9784, -76.4922),
    "Baltimore City": (39.2904, -76.6122),
    "Bel Air": (39.5359, -76.3483),
    "Columbia": (39.2037, -76.8610),
    "Essex": (39.3045, -76.4683),
    "Frederick": (39.4143, -77.4105),
    "Gaithersburg": (39.1434, -77.2014),
    "Glen Burnie": (39.1626, -76.6247),
    "Largo": (38.9058, -76.8430),
    "Parkville": (39.3777, -76.5400),
    "White Oak": (39.0384, -76.9903),
}

# Approximate interstate/highway hints for natural-language origin inference
HIGHWAY_HINTS = {
    "i-95": [
        {"label": "I-95 near White Marsh", "lat": 39.3868, "lon": -76.4433},
        {"label": "I-95 near Baltimore", "lat": 39.2904, "lon": -76.6122},
        {"label": "I-95 near Laurel", "lat": 39.0993, "lon": -76.8483},
    ],
    "i-695": [
        {"label": "I-695 east near Parkville", "lat": 39.3777, "lon": -76.5400},
        {"label": "I-695 west near Woodlawn", "lat": 39.3151, "lon": -76.7341},
        {"label": "I-695 south near Glen Burnie", "lat": 39.1626, "lon": -76.6247},
    ],
    "i-495": [
        {"label": "I-495 near Largo", "lat": 38.9058, "lon": -76.8430},
        {"label": "I-495 near Silver Spring", "lat": 39.0384, "lon": -76.9903},
    ],
    "i-270": [
        {"label": "I-270 near Gaithersburg", "lat": 39.1434, "lon": -77.2014},
        {"label": "I-270 near Frederick", "lat": 39.4143, "lon": -77.4105},
    ],
    "i-70": [
        {"label": "I-70 near Columbia", "lat": 39.2037, "lon": -76.8610},
        {"label": "I-70 near Frederick", "lat": 39.4143, "lon": -77.4105},
    ],
    "i-83": [
        {"label": "I-83 near Baltimore", "lat": 39.3770, "lon": -76.6460},
    ],
    "i-81": [
        {"label": "I-81 near Hagerstown", "lat": 39.6418, "lon": -77.7200},
    ],
    "i-97": [
        {"label": "I-97 near Glen Burnie", "lat": 39.1626, "lon": -76.6247},
        {"label": "I-97 near Annapolis", "lat": 38.9784, "lon": -76.4922},
    ],
    "i-295": [
        {"label": "I-295 near Baltimore-Washington corridor", "lat": 39.1450, "lon": -76.7850},
        {"label": "I-295 near DC side", "lat": 38.8790, "lon": -76.9660},
    ],
    "i-795": [
        {"label": "I-795 near Owings Mills", "lat": 39.4070, "lon": -76.7827},
    ],
    "i-895": [
        {"label": "I-895 near Baltimore Harbor Tunnel", "lat": 39.2380, "lon": -76.5690},
    ],
}

HIGHWAY_ALIASES = {
    "i95": "i-95",
    "i-95": "i-95",
    "95": "i-95",
    "interstate95": "i-95",
    "interstate-95": "i-95",

    "i695": "i-695",
    "i-695": "i-695",
    "695": "i-695",
    "interstate695": "i-695",
    "interstate-695": "i-695",
    "baltimore beltway": "i-695",

    "i495": "i-495",
    "i-495": "i-495",
    "495": "i-495",
    "interstate495": "i-495",
    "interstate-495": "i-495",
    "capital beltway": "i-495",

    "i270": "i-270",
    "i-270": "i-270",
    "270": "i-270",
    "interstate270": "i-270",
    "interstate-270": "i-270",

    "i70": "i-70",
    "i-70": "i-70",
    "70": "i-70",
    "interstate70": "i-70",
    "interstate-70": "i-70",

    "i83": "i-83",
    "i-83": "i-83",
    "83": "i-83",
    "interstate83": "i-83",
    "interstate-83": "i-83",

    "i81": "i-81",
    "i-81": "i-81",
    "81": "i-81",
    "interstate81": "i-81",
    "interstate-81": "i-81",

    "i97": "i-97",
    "i-97": "i-97",
    "97": "i-97",
    "interstate97": "i-97",
    "interstate-97": "i-97",

    "i295": "i-295",
    "i-295": "i-295",
    "295": "i-295",
    "interstate295": "i-295",
    "interstate-295": "i-295",
    "bw parkway": "i-295",
    "baltimore-washington parkway": "i-295",

    "i795": "i-795",
    "i-795": "i-795",
    "795": "i-795",

    "i895": "i-895",
    "i-895": "i-895",
    "895": "i-895",
}

cust_cols = [c for c in df.columns if "Customers Served" in c]
wait_cols = [c for c in df.columns if "Wait Time" in c]

for c in cust_cols:
    df[c] = (
        df[c].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in wait_cols:
    df[c] = (
        df[c].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

for period in ["FY23", "FY24", "FY25"]:
    df[f"{period} Efficiency"] = df[f"{period} Customers Served"] / df[f"{period} Wait Time"]

for m in MONTHS:
    df[f"{m} Efficiency"] = df[f"{m} Customers Served"] / df[f"{m} Wait Time"]


# -------------------------
# helpers
# -------------------------
def _safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def df_to_text(d: pd.DataFrame, max_rows: int = 8) -> str:
    return d.head(max_rows).to_string(index=False)


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


def _filter_by_region(data: pd.DataFrame, region: Optional[str]) -> pd.DataFrame:
    """
    Keyword filter on Branch name only.
    This is useful for direct branch-related queries like 'Baltimore' or 'Largo',
    but not for highway origin inference.
    """
    if not region:
        return data
    region = region.strip().lower()
    if not region:
        return data
    return data[data["Branch"].astype(str).str.lower().str.contains(region, na=False)]


# -------------------------
# analytics tools
# -------------------------
def top_longest_wait(month: str = "December 2025", n: int = 5):
    return df[["Branch", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
        by=f"{month} Wait Time", ascending=False
    ).head(n)


def top_shortest_wait(month: str = "December 2025", n: int = 5, region: Optional[str] = None):
    tmp = _filter_by_region(df, region)
    return tmp[["Branch", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
        by=f"{month} Wait Time", ascending=True
    ).head(n)


def top_best_efficiency(month: str = "December 2025", n: int = 5):
    return df[["Branch", f"{month} Efficiency", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
        by=f"{month} Efficiency", ascending=False
    ).head(n)


def biggest_wait_increase(from_period: str = "FY25", to_month: str = "December 2025", n: int = 5):
    tmp = df.copy()
    tmp["Wait Change"] = tmp[f"{to_month} Wait Time"] - tmp[f"{from_period} Wait Time"]
    return tmp[["Branch", f"{from_period} Wait Time", f"{to_month} Wait Time", "Wait Change"]].sort_values(
        by="Wait Change", ascending=False
    ).head(n)


def branch_summary(branch_name: str):
    row = df[df["Branch"].astype(str).str.lower() == (branch_name or "").strip().lower()]
    if row.empty:
        return {"error": f"Branch '{branch_name}' not found."}

    r = row.iloc[0]
    return {
        "Branch": r["Branch"],
        "FY23": {
            "served": _safe_int(r["FY23 Customers Served"]),
            "wait": _safe_float(r["FY23 Wait Time"]),
            "eff": _safe_float(r["FY23 Efficiency"]),
        },
        "FY24": {
            "served": _safe_int(r["FY24 Customers Served"]),
            "wait": _safe_float(r["FY24 Wait Time"]),
            "eff": _safe_float(r["FY24 Efficiency"]),
        },
        "FY25": {
            "served": _safe_int(r["FY25 Customers Served"]),
            "wait": _safe_float(r["FY25 Wait Time"]),
            "eff": _safe_float(r["FY25 Efficiency"]),
        },
        "Monthly_2025": {
            m: {
                "served": _safe_int(r[f"{m} Customers Served"]),
                "wait": _safe_float(r[f"{m} Wait Time"]),
                "eff": _safe_float(r[f"{m} Efficiency"]),
            }
            for m in MONTHS
        },
    }


# -------------------------
# Mapbox tools
# -------------------------
def geocode_address(address: str):
    """Convert address/place to coordinates using Mapbox geocoding."""
    if not address:
        return None

    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{quote(address, safe='')}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "limit": 1,
    }

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None

    data = r.json()
    features = data.get("features", [])
    if not features:
        return None

    lon, lat = features[0]["center"]
    return lat, lon


def traffic_eta_minutes(origin_lat: float, origin_lon: float, branch_name: str):
    """Returns current traffic ETA (minutes) from origin -> branch using Mapbox driving-traffic."""
    if not branch_name:
        return {"error": "Missing branch name."}

    key_map = {k.lower(): k for k in BRANCH_COORDS.keys()}
    bkey = key_map.get(branch_name.strip().lower())
    if not bkey:
        return {"error": f"No coordinates found for branch '{branch_name}'. Available: {list(BRANCH_COORDS.keys())}"}

    dest_lat, dest_lon = BRANCH_COORDS[bkey]

    url = (
        "https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    )
    params = {
        "access_token": MAPBOX_TOKEN,
        "overview": "false",
        "alternatives": "false",
        "geometries": "geojson",
    }

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return {"error": f"Mapbox error {r.status_code}: {r.text[:200]}"}

    data = r.json()
    routes = data.get("routes", [])
    if not routes:
        return {"error": "No route returned from Mapbox."}

    duration_sec = routes[0]["duration"]
    return {
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "branch": bkey,
        "eta_minutes": round(duration_sec / 60, 1),
    }


def branch_wait_time(branch_name: str, month: str = "December 2025"):
    if not branch_name:
        return {"error": "Missing branch name."}

    if month not in MONTHS:
        month = "December 2025"

    row = df[df["Branch"].astype(str).str.lower() == branch_name.strip().lower()]
    if row.empty:
        return {"error": f"Branch '{branch_name}' not found."}

    val = row.iloc[0][f"{month} Wait Time"]
    try:
        wait = float(val)
    except Exception:
        wait = None

    return {
        "branch": row.iloc[0]["Branch"],
        "month": month,
        "wait_minutes": wait,
    }


def best_branch_by_total_time(origin_lat: float, origin_lon: float, month: str = "December 2025"):
    results = []

    for branch_name in BRANCH_COORDS.keys():
        traffic = traffic_eta_minutes(origin_lat, origin_lon, branch_name)
        if "error" in traffic:
            continue

        wait_info = branch_wait_time(branch_name, month=month)
        wait_min = wait_info.get("wait_minutes")
        if wait_min is None:
            continue

        total_min = round(float(traffic["eta_minutes"]) + float(wait_min), 1)

        results.append({
            "branch": branch_name,
            "drive_eta_minutes": traffic["eta_minutes"],
            "historical_wait_minutes": wait_min,
            "wait_month": month,
            "estimated_total_minutes": total_min,
        })

    if not results:
        return {"error": "Could not calculate total time for any branch."}

    results = sorted(results, key=lambda x: x["estimated_total_minutes"])
    return {
        "best_branch": results[0],
        "all_ranked_branches": results,
    }


def best_branch_with_delay(
    origin_lat: float,
    origin_lon: float,
    extra_delay_minutes: float = 0,
    month: str = "December 2025",
):
    results = []

    for branch_name in BRANCH_COORDS.keys():
        traffic = traffic_eta_minutes(origin_lat, origin_lon, branch_name)
        if "error" in traffic:
            continue

        wait_info = branch_wait_time(branch_name, month=month)
        wait_min = wait_info.get("wait_minutes")
        if wait_min is None:
            continue

        adjusted_drive = round(float(traffic["eta_minutes"]) + float(extra_delay_minutes), 1)
        total_min = round(adjusted_drive + float(wait_min), 1)

        results.append({
            "branch": branch_name,
            "drive_eta_minutes": traffic["eta_minutes"],
            "extra_delay_minutes": float(extra_delay_minutes),
            "adjusted_drive_minutes": adjusted_drive,
            "historical_wait_minutes": wait_min,
            "wait_month": month,
            "estimated_total_minutes": total_min,
        })

    if not results:
        return {"error": "Could not calculate total time for any branch."}

    results = sorted(results, key=lambda x: x["estimated_total_minutes"])
    return {
        "best_branch": results[0],
        "all_ranked_branches": results,
    }


# -------------------------
# natural language extraction
# -------------------------
DELAY_RE = re.compile(
    r"(\d+)\s*(?:minute|minutes|min)\s*(?:traffic jam|delay|extra traffic|jam)",
    re.I,
)

COORD_RE = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)")

HIGHWAY_RE = re.compile(
    r"\b(i[\-\s]?\d{2,3}|interstate[\-\s]?\d{2,3}|capital beltway|baltimore beltway|bw parkway|baltimore-washington parkway)\b",
    re.I,
)

DIRECTION_RE = re.compile(
    r"\b(north|south|east|west|northbound|southbound|eastbound|westbound)\b",
    re.I,
)

NEAR_RE = re.compile(r"\bnear\s+([a-zA-Z0-9\-\s]+)", re.I)
MIDDLE_RE = re.compile(r"\bmiddle of\b", re.I)


def extract_delay_from_text(text: str):
    m = DELAY_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1))


def extract_coords_from_text(text: str):
    """
    Returns (lat, lon) if the user typed something like '39.29,-76.61'
    """
    m = COORD_RE.search(text or "")
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    return lat, lon


def normalize_highway_name(text: str) -> Optional[str]:
    m = HIGHWAY_RE.search(text or "")
    if not m:
        return None

    raw = m.group(1).strip().lower()
    raw_no_spaces = raw.replace(" ", "").replace("interstate", "i")

    if raw_no_spaces.startswith("i") and "-" not in raw_no_spaces and len(raw_no_spaces) > 1:
        raw_no_spaces = raw_no_spaces[0] + "-" + raw_no_spaces[1:]

    return HIGHWAY_ALIASES.get(raw_no_spaces) or HIGHWAY_ALIASES.get(raw)


def infer_origin_from_highway_text(user_text: str):
    highway = normalize_highway_name(user_text)
    if not highway:
        return None

    candidates = HIGHWAY_HINTS.get(highway, [])
    if not candidates:
        return None

    near_match = NEAR_RE.search(user_text or "")
    direction_match = DIRECTION_RE.search(user_text or "")
    middle_match = MIDDLE_RE.search(user_text or "")

    # If user says "near X", try to geocode that place first
    if near_match:
        near_place = near_match.group(1).strip(" ,.")
        coords = geocode_address(f"{near_place}, Maryland")
        if coords:
            return {
                "source": "highway_near_place",
                "highway": highway,
                "lat": coords[0],
                "lon": coords[1],
                "label": f"{highway} near {near_place}",
            }

    # If user says "middle of", use middle candidate
    if middle_match and candidates:
        mid = candidates[len(candidates) // 2]
        return {
            "source": "highway_midpoint_hint",
            "highway": highway,
            "lat": mid["lat"],
            "lon": mid["lon"],
            "label": mid["label"],
        }

    # Direction-based guess
    if direction_match and candidates:
        d = direction_match.group(1).lower()
        if d in ("north", "northbound", "east", "eastbound"):
            chosen = candidates[0]
        else:
            chosen = candidates[-1]

        return {
            "source": "highway_direction_hint",
            "highway": highway,
            "lat": chosen["lat"],
            "lon": chosen["lon"],
            "label": chosen["label"],
        }

    # Default to middle candidate
    chosen = candidates[len(candidates) // 2]
    return {
        "source": "highway_default_hint",
        "highway": highway,
        "lat": chosen["lat"],
        "lon": chosen["lon"],
        "label": chosen["label"],
    }


# -------------------------
# LLM router
# -------------------------
ROUTER_INSTRUCTIONS = f"""
Return ONLY valid JSON (no markdown).

Schema:
{{
  "action": one of ["top_longest_wait","top_shortest_wait","top_best_efficiency",
                    "biggest_wait_increase","branch_summary","traffic_eta",
                    "travel_plus_wait","best_branch_total_time","best_branch_with_delay","help"],
  "month": one of {MONTHS} or null,
  "n": integer or null,
  "branch": string or null,
  "region": string or null,
  "origin_lat": number or null,
  "origin_lon": number or null,
  "extra_delay_minutes": number or null
}}

Rules:
- If user asks which branch to go to and mentions an extra traffic jam / delay / added traffic minutes -> best_branch_with_delay
- If user asks "best branch", "fastest branch", "least total time", "shortest total time", "best option for me" considering traffic and wait -> best_branch_total_time
- If user asks travel time and also mentions wait time / including wait / total time -> travel_plus_wait
- If user asks only "traffic", "ETA", "drive time", "travel time" -> traffic_eta
- If user asks "longest wait", "highest wait", "worst wait" -> top_longest_wait
- If user asks "shortest wait", "lowest wait", "least wait", "minimum wait" -> top_shortest_wait
- If user asks "best efficiency", "most efficient" -> top_best_efficiency
- If user asks "increase", "got worse" -> biggest_wait_increase
- If user asks "summary" or clearly names a single branch -> branch_summary
- If user gives coordinates, set origin_lat/origin_lon
- If user provides an address/place, put it in region
- If user mentions a highway/interstate/beltway/parkway as current location, put that phrase in region
- If user does not specify month, set month=null
- If user does not specify n, set n=null
- If unclear -> help

Examples:
User: "Traffic ETA from 39.29,-76.61 to Largo"
-> {{"action":"traffic_eta","branch":"Largo","origin_lat":39.29,"origin_lon":-76.61,"month":null,"n":null,"region":null,"extra_delay_minutes":null}}

User: "Traffic ETA from Baltimore, MD to Largo"
-> {{"action":"traffic_eta","branch":"Largo","origin_lat":null,"origin_lon":null,"month":null,"n":null,"region":"Baltimore, MD","extra_delay_minutes":null}}

User: "How long from 2907 Fallstaff Road, Baltimore, MD to Largo including historical wait time?"
-> {{"action":"travel_plus_wait","branch":"Largo","region":"2907 Fallstaff Road, Baltimore, MD","month":null,"n":null,"origin_lat":null,"origin_lon":null,"extra_delay_minutes":null}}

User: "Which MVA branch is best for me from Baltimore, MD considering traffic and wait time?"
-> {{"action":"best_branch_total_time","branch":null,"region":"Baltimore, MD","origin_lat":null,"origin_lon":null,"month":null,"n":null,"extra_delay_minutes":null}}

User: "I'm on I-95 south, which branch should I go to?"
-> {{"action":"best_branch_total_time","branch":null,"region":"I-95 south","origin_lat":null,"origin_lon":null,"month":null,"n":null,"extra_delay_minutes":null}}

User: "I'm on I-695 near Parkville and there is a 10 minute delay. Which branch should I go to?"
-> {{"action":"best_branch_with_delay","branch":null,"region":"I-695 near Parkville","origin_lat":null,"origin_lon":null,"month":null,"n":null,"extra_delay_minutes":10}}
"""


def route_intent(user_text: str) -> dict:
    fallback = {
        "action": "help",
        "month": None,
        "n": None,
        "branch": None,
        "region": None,
        "origin_lat": None,
        "origin_lon": None,
        "extra_delay_minutes": None,
        "location_source": None,
    }

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{ROUTER_INSTRUCTIONS}\n\nUser: {user_text}",
        )
        raw = (resp.text or "").strip()
        cmd = json.loads(_extract_json(raw))
    except Exception:
        cmd = fallback.copy()

    # Ensure keys exist
    for k, v in fallback.items():
        cmd.setdefault(k, v)

    # Extract delay from text if router missed it
    if cmd.get("extra_delay_minutes") is None:
        delay = extract_delay_from_text(user_text)
        if delay is not None:
            cmd["extra_delay_minutes"] = delay

    # Extract coordinates from text if router missed them
    if cmd.get("action") in ("traffic_eta", "travel_plus_wait", "best_branch_total_time", "best_branch_with_delay"):
        if cmd.get("origin_lat") is None or cmd.get("origin_lon") is None:
            coords = extract_coords_from_text(user_text)
            if coords:
                cmd["origin_lat"], cmd["origin_lon"] = coords
                cmd["location_source"] = "typed_coordinates"

    # Highway/interstate approximation if still no coords
    if cmd.get("action") in ("traffic_eta", "travel_plus_wait", "best_branch_total_time", "best_branch_with_delay"):
        if cmd.get("origin_lat") is None or cmd.get("origin_lon") is None:
            approx = infer_origin_from_highway_text(user_text)
            if approx:
                cmd["origin_lat"] = approx["lat"]
                cmd["origin_lon"] = approx["lon"]
                cmd["region"] = approx["label"]
                cmd["location_source"] = approx["source"]

    return cmd


def explain(user_text: str, tool_result_text: str) -> str:
    prompt = f"""You are a public-service operations analyst.
Use 3 to 6 bullets maximum.
Mention key branches and values.
End with 1 practical recommendation.

User question:
{user_text}

Data result:
{tool_result_text}
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (resp.text or "").strip()
    except Exception:
        return tool_result_text


# -------------------------
# command execution
# -------------------------
def resolve_origin(cmd: dict):
    """
    Resolve origin for traffic-based actions.
    Priority:
    1) explicit origin_lat/origin_lon from payload or text
    2) region/address geocoding
    """
    origin_lat = cmd.get("origin_lat")
    origin_lon = cmd.get("origin_lon")
    region = (cmd.get("region") or "").strip()
    location_source = cmd.get("location_source")

    if origin_lat is not None and origin_lon is not None:
        return {
            "origin_lat": float(origin_lat),
            "origin_lon": float(origin_lon),
            "location_source": location_source or "provided_coordinates",
            "origin_label": region or None,
        }

    if region:
        coords = geocode_address(region)
        if coords:
            return {
                "origin_lat": float(coords[0]),
                "origin_lon": float(coords[1]),
                "location_source": location_source or "geocoded_address",
                "origin_label": region,
            }

        return {"error": f"Could not locate address/origin '{region}'. Try a more specific place."}

    return {"error": "Please provide an origin address, highway phrase, browser location, or coordinates."}


def run_tool(cmd: dict):
    action = cmd.get("action")
    month = cmd.get("month") or "December 2025"
    n = int(cmd.get("n") or 5)
    branch = cmd.get("branch")
    region = cmd.get("region")

    if month not in MONTHS:
        month = "December 2025"

    # --- traffic ETA only ---
    if action == "traffic_eta":
        if not branch:
            return {"help": "Please provide a destination branch, for example: 'Traffic ETA to Largo'."}

        resolved = resolve_origin(cmd)
        if "error" in resolved:
            return {"help": resolved["error"]}

        traffic = traffic_eta_minutes(resolved["origin_lat"], resolved["origin_lon"], branch)
        if "error" in traffic:
            return {"traffic": traffic, "table_text": json.dumps(traffic, indent=2)}

        result = {
            "branch": traffic["branch"],
            "eta_minutes": traffic["eta_minutes"],
            "origin": traffic["origin"],
            "origin_label": resolved.get("origin_label"),
            "location_source": resolved.get("location_source"),
        }
        return {"traffic": result, "table_text": json.dumps(result, indent=2)}

    # --- drive ETA + historical wait ---
    if action == "travel_plus_wait":
        if not branch:
            return {"help": "Please provide a destination branch, for example: 'How long to Largo including wait time?'."}

        resolved = resolve_origin(cmd)
        if "error" in resolved:
            return {"help": resolved["error"]}

        traffic = traffic_eta_minutes(resolved["origin_lat"], resolved["origin_lon"], branch)
        if "error" in traffic:
            return {"travel_plus_wait": {"traffic": traffic}, "table_text": json.dumps(traffic, indent=2)}

        wait_info = branch_wait_time(branch, month=month)
        if "error" in wait_info:
            result = {"traffic": traffic, "wait": wait_info}
            return {"travel_plus_wait": result, "table_text": json.dumps(result, indent=2)}

        wait_min = wait_info.get("wait_minutes")
        total_min = None
        if wait_min is not None:
            total_min = round(float(traffic["eta_minutes"]) + float(wait_min), 1)

        result = {
            "origin_label": resolved.get("origin_label"),
            "location_source": resolved.get("location_source"),
            "destination_branch": wait_info["branch"],
            "drive_eta_minutes": traffic["eta_minutes"],
            "historical_wait_minutes": wait_min,
            "wait_month": wait_info["month"],
            "estimated_total_minutes": total_min,
        }
        return {"travel_plus_wait": result, "table_text": json.dumps(result, indent=2)}

    # --- best branch with extra delay ---
    if action == "best_branch_with_delay":
        resolved = resolve_origin(cmd)
        if "error" in resolved:
            return {"help": resolved["error"]}

        extra_delay = float(cmd.get("extra_delay_minutes") or 0)
        result = best_branch_with_delay(
            resolved["origin_lat"],
            resolved["origin_lon"],
            extra_delay_minutes=extra_delay,
            month=month,
        )

        if "error" not in result:
            result["origin_label"] = resolved.get("origin_label")
            result["location_source"] = resolved.get("location_source")

        return {"best_branch_with_delay": result, "table_text": json.dumps(result, indent=2)}

    # --- best branch by drive + wait ---
    if action == "best_branch_total_time":
        resolved = resolve_origin(cmd)
        if "error" in resolved:
            return {"help": resolved["error"]}

        result = best_branch_by_total_time(
            resolved["origin_lat"],
            resolved["origin_lon"],
            month=month,
        )

        if "error" not in result:
            result["origin_label"] = resolved.get("origin_label")
            result["location_source"] = resolved.get("location_source")

        return {"best_branch_total_time": result, "table_text": json.dumps(result, indent=2)}

    # --- reporting queries ---
    if action == "top_longest_wait":
        d = _filter_by_region(df, region)
        d = d[["Branch", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
            by=f"{month} Wait Time", ascending=False
        ).head(n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}

    if action == "top_shortest_wait":
        d = top_shortest_wait(month, n, region=region)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}

    if action == "top_best_efficiency":
        d = _filter_by_region(df, region)
        d = d[["Branch", f"{month} Efficiency", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
            by=f"{month} Efficiency", ascending=False
        ).head(n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}

    if action == "biggest_wait_increase":
        tmp = _filter_by_region(df.copy(), region)
        tmp["Wait Change"] = tmp[f"{month} Wait Time"] - tmp["FY25 Wait Time"]
        d = tmp[["Branch", "FY25 Wait Time", f"{month} Wait Time", "Wait Change"]].sort_values(
            by="Wait Change", ascending=False
        ).head(n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}

    if action == "branch_summary":
        s = branch_summary(branch or "")
        return {"summary": s, "table_text": json.dumps(s, indent=2)}

    return {
        "help": (
            "Try: 'Lowest wait time in December 2025', "
            "'Summary: Largo', "
            "'Traffic ETA from Baltimore to Largo', or "
            "'I'm on I-95 south, which branch should I go to?'"
        )
    }


# -------------------------
# API
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None


@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(payload: ChatIn):
    cmd = route_intent(payload.message)

    # Browser geolocation overrides inferred origin
    if payload.origin_lat is not None and payload.origin_lon is not None:
        cmd["origin_lat"] = payload.origin_lat
        cmd["origin_lon"] = payload.origin_lon
        cmd["location_source"] = "browser_geolocation"

    tool_out = run_tool(cmd)

    if "help" in tool_out:
        return {
            "command": cmd,
            "answer": tool_out["help"],
            "data": tool_out,
        }

    # traffic only
    if "traffic" in tool_out:
        t = tool_out["traffic"]
        if "error" in t:
            return {"command": cmd, "answer": t["error"], "data": tool_out}

        source_note = ""
        if t.get("location_source") and t["location_source"] != "browser_geolocation":
            source_note = f" Location source: {t['location_source']}."

        return {
            "command": cmd,
            "answer": f"Current driving ETA to {t['branch']}: {t['eta_minutes']} minutes.{source_note}",
            "data": tool_out,
        }

    # travel + wait
    if "travel_plus_wait" in tool_out:
        r = tool_out["travel_plus_wait"]

        if isinstance(r, dict) and "traffic" in r and isinstance(r["traffic"], dict) and "error" in r["traffic"]:
            return {"command": cmd, "answer": r["traffic"]["error"], "data": tool_out}

        total = r.get("estimated_total_minutes")
        source_note = ""
        if r.get("location_source") and r["location_source"] != "browser_geolocation":
            source_note = f" Estimated from {r['location_source']}."

        if total is None:
            return {
                "command": cmd,
                "answer": (
                    f"Drive ETA to {r['destination_branch']}: {r['drive_eta_minutes']} minutes. "
                    f"No wait-time value was available for {r['wait_month']}.{source_note}"
                ),
                "data": tool_out,
            }

        return {
            "command": cmd,
            "answer": (
                f"Estimated total time to {r['destination_branch']} is about {total} minutes "
                f"({r['drive_eta_minutes']} minutes driving now + "
                f"{r['historical_wait_minutes']} minutes historical wait for {r['wait_month']})."
                f"{source_note}"
            ),
            "data": tool_out,
        }

    # best branch total time
    if "best_branch_total_time" in tool_out:
        r = tool_out["best_branch_total_time"]

        if "error" in r:
            return {"command": cmd, "answer": r["error"], "data": tool_out}

        best = r["best_branch"]
        source_note = ""
        if r.get("location_source") and r["location_source"] != "browser_geolocation":
            source_note = f" This was estimated from {r['location_source']}."

        return {
            "command": cmd,
            "answer": (
                f"Best branch right now is {best['branch']} with an estimated total time of "
                f"{best['estimated_total_minutes']} minutes "
                f"({best['drive_eta_minutes']} minutes driving + "
                f"{best['historical_wait_minutes']} minutes wait in {best['wait_month']})."
                f"{source_note}"
            ),
            "data": tool_out,
        }

    # best branch with delay
    if "best_branch_with_delay" in tool_out:
        r = tool_out["best_branch_with_delay"]

        if "error" in r:
            return {"command": cmd, "answer": r["error"], "data": tool_out}

        best = r["best_branch"]
        source_note = ""
        if r.get("location_source") and r["location_source"] != "browser_geolocation":
            source_note = f" This was estimated from {r['location_source']}."

        return {
            "command": cmd,
            "answer": (
                f"Given the extra {best['extra_delay_minutes']} minute traffic delay, "
                f"the best branch is {best['branch']} with an estimated total time of "
                f"{best['estimated_total_minutes']} minutes "
                f"({best['adjusted_drive_minutes']} minutes driving including delay + "
                f"{best['historical_wait_minutes']} minutes wait in {best['wait_month']})."
                f"{source_note}"
            ),
            "data": tool_out,
        }

    # everything else
    answer = explain(payload.message, tool_out.get("table_text", ""))
    return {
        "command": cmd,
        "answer": answer,
        "data": tool_out,
    }
