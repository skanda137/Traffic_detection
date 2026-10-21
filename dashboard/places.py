import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from feature_pipeline import haversine_np  # noqa: E402

RANGES_PATH = ROOT / "reports" / "input_ranges.json"

# Approximate Bengaluru coordinates, for demo use only
CANDIDATE_PLACES = {
    "Majestic": (12.9767, 77.5726), "MG Road": (12.9756, 77.6068), "Shivajinagar": (12.9857, 77.6057),
    "Cubbon Park": (12.9763, 77.5929), "Vidhana Soudha": (12.9796, 77.5907), "Richmond Circle": (12.9650, 77.5970),
    "KR Market": (12.9648, 77.5780), "Lalbagh": (12.9507, 77.5848), "Basavanagudi": (12.9422, 77.5736),
    "Jayanagar": (12.9250, 77.5938), "JP Nagar": (12.9063, 77.5857), "Banashankari": (12.9255, 77.5468),
    "Vijayanagar": (12.9712, 77.5370), "Rajajinagar": (12.9916, 77.5520), "Malleshwaram": (13.0035, 77.5643),
    "Yeshwanthpur": (13.0285, 77.5400), "Peenya": (13.0285, 77.5197), "RT Nagar": (13.0210, 77.5950),
    "Hebbal": (13.0358, 77.5970), "Yelahanka": (13.1007, 77.5963), "Hennur": (13.0360, 77.6390),
    "Banaswadi": (13.0143, 77.6510), "Indiranagar": (12.9784, 77.6408), "Ulsoor": (12.9822, 77.6200),
    "Domlur": (12.9610, 77.6387), "Koramangala": (12.9352, 77.6245), "Silk Board": (12.9177, 77.6233),
    "BTM Layout": (12.9166, 77.6101), "HSR Layout": (12.9116, 77.6474), "Bellandur": (12.9304, 77.6784),
    "Marathahalli": (12.9569, 77.7011), "Tin Factory": (12.9985, 77.6706), "KR Puram": (13.0085, 77.6960),
    "Whitefield": (12.9698, 77.7500), "Electronic City": (12.8452, 77.6602), "Kengeri": (12.9174, 77.4833),
    "Airport": (13.1986, 77.7066),
}

with open(RANGES_PATH) as f:
    RANGES = json.load(f)

# Keep only places inside the training rows' lat/lon box
PLACES = {name: (lat, lon) for name, (lat, lon) in CANDIDATE_PLACES.items()
          if RANGES["lat_min"] <= lat <= RANGES["lat_max"] and RANGES["lon_min"] <= lon <= RANGES["lon_max"]}
DROPPED_PLACES = sorted(set(CANDIDATE_PLACES) - set(PLACES))


def route_from_places(src, dst):
    """Place names -> route fields. distance = straight-line km x training median tortuosity, so the
    tortuosity range check passes by construction; distance/haversine range checks still apply."""
    if src == dst:
        raise ValueError(f"From and To are the same place ({src}); pick two different places")
    (lat_src, lon_src), (lat_dest, lon_dest) = PLACES[src], PLACES[dst]
    haversine_km = float(haversine_np(lat_src, lon_src, lat_dest, lon_dest))
    return {"lat_src": lat_src, "lon_src": lon_src, "lat_dest": lat_dest, "lon_dest": lon_dest,
            "distance": haversine_km * RANGES["raw_tortuosity_median"]}


if __name__ == "__main__":
    print(f"{len(PLACES)} of {len(CANDIDATE_PLACES)} places kept; dropped: {DROPPED_PLACES or 'none'}")
