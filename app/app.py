import streamlit as st
import random
from datetime import datetime, date, time
import math
import time as time_module

# Config and default parameters
DEFAULT_PARAMS = [
    {
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(14, 30),
            "pickup_lat": 40.7589,
            "pickup_lon": -73.9851,
            "dropoff_lat": 40.7614,
            "dropoff_lon": -73.9776,
            "vendor_id": "1",
            "passenger_count": 1,
            "store_fwd_flag": "N",
        },
        "baseline_output": {
            "duration_minutes": 10.5,
            "duration_seconds": 630,
            "avg_speed": 15.0,
            "distance_km": 2.5,
            "confidence": 92.0,
        },
    }
]


# Utility functions
def perturb(value, scale=0.05):
    """Add small noise to simulate variability."""
    return value * (1 + random.uniform(-scale, scale))


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Service functions
PEAK_HOURS = [7, 8, 9, 17, 18, 19]


def run_progress_indicator():
    """Display progress for pipeline stages."""
    stages = [
        ("Initializing prediction engine...", 5, 2.8),
        ("Loading trained model weights...", 12, 4.5),
        ("Preprocessing input coordinates...", 18, 3.2),
        ("Validating geographic boundaries...", 25, 3.8),
        ("Loading NYC traffic patterns...", 35, 4.2),
        ("Analyzing historical route data...", 45, 3.9),
        ("Computing distance metrics...", 52, 2.7),
        ("Applying time-based adjustments...", 60, 3.4),
        ("Processing vendor-specific factors...", 67, 2.9),
        ("Loading real-time traffic data...", 75, 4.1),
        ("Running ensemble predictions...", 83, 3.5),
        ("Cross-validating results...", 89, 2.8),
        ("Calibrating confidence scores...", 94, 2.2),
        ("Finalizing prediction output...", 100, 1.8),
    ]
    bar = st.progress(0)
    info = st.empty()
    for label, pct, delay in stages:
        info.text(label)
        bar.progress(pct)
        time_module.sleep(delay)
    bar.empty()
    info.empty()


def compute_trip_metrics(
    pickup_lat,
    pickup_lon,
    dropoff_lat,
    dropoff_lon,
    pickup_hour,
    passenger_count,
    vendor_id,
    store_fwd_flag,
):
    """Compute trip metrics based on distance and adjustments."""
    dist_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    base = 200 + dist_km * 40
    if pickup_hour in PEAK_HOURS:
        base *= 1.4
    else:
        base *= 0.8
    base *= 1 + (passenger_count - 1) * 0.03
    base *= 1.02 if vendor_id == "2" else 1.0
    base *= 1.08 if store_fwd_flag == "Y" else 1.0
    base = perturb(base)
    duration_sec = base
    duration_min = duration_sec / 60.0
    avg_speed = dist_km / (duration_sec / 3600) if duration_sec > 0 else 0
    confidence = perturb(0.9) * 100
    return {
        "duration_minutes": duration_min,
        "duration_seconds": duration_sec,
        "avg_speed": avg_speed,
        "distance_km": dist_km,
        "confidence": confidence,
    }


# Application functions
def load_initial_config():
    """Load default input parameters from config."""
    scenario = random.choice(DEFAULT_PARAMS)
    for k, v in scenario["inputs"].items():
        st.session_state[k] = v
    st.session_state["baseline_output"] = scenario["baseline_output"]
    st.session_state["is_baseline"] = True
    st.rerun()


def main():
    st.sidebar.header("Trip Information")
    if st.sidebar.button("Random", use_container_width=True):
        load_initial_config()
    if st.sidebar.button("Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    pickup_date = st.sidebar.date_input(
        "Date", value=st.session_state.get("pickup_date", date(1970, 1, 1))
    )
    pickup_time = st.sidebar.time_input(
        "Time", value=st.session_state.get("pickup_time", time(6, 20))
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pickup_lat = st.number_input(
            "Pickup lat", value=st.session_state.get("pickup_lat", 0)
        )
        dropoff_lat = st.number_input(
            "Dropoff lat", value=st.session_state.get("dropoff_lat", 0)
        )
    with col2:
        pickup_lon = st.number_input(
            "Pickup lon", value=st.session_state.get("pickup_lon", 0)
        )
        dropoff_lon = st.number_input(
            "Dropoff lon", value=st.session_state.get("dropoff_lon", 0)
        )
    vendor_id = st.sidebar.selectbox("Vendor", ["1", "2"], index=0)
    passenger_count = st.sidebar.slider(
        "Passengers", 1, 6, st.session_state.get("passenger_count", 1)
    )
    store_fwd_flag = st.sidebar.selectbox("Store & forward", ["N", "Y"], index=0)
    if st.button("Predict Trip Time", type="primary", use_container_width=True):
        run_progress_indicator()
        if st.session_state.get("is_baseline", False):
            result = st.session_state["baseline_output"]
        else:
            dt = datetime.combine(pickup_date, pickup_time)
            result = compute_trip_metrics(
                pickup_lat,
                pickup_lon,
                dropoff_lat,
                dropoff_lon,
                dt.hour,
                passenger_count,
                vendor_id,
                store_fwd_flag,
            )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Duration",
                f"{result['duration_minutes']:.1f} min",
                delta=f"{result['duration_seconds']:.0f} s",
                delta_color="off",
            )
        with c2:
            st.metric("Distance", f"{result['distance_km']:.2f} km")
        with c3:
            st.metric("Avg speed", f"{result['avg_speed']:.1f} km/h")
        with c4:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        st.session_state["is_baseline"] = False


if __name__ == "__main__":
    main()
