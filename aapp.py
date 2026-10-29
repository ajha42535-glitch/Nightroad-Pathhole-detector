
import math
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "yolo26_best.pt"
REPORT_DIR = APP_DIR / "reports"
REPORT_FILE = REPORT_DIR / "pothole_reports.csv"

MODEL_URL = (
    "https://huggingface.co/DanielsStulpe/"
    "pothole-detection/resolve/main/yolo26_best.pt"
)

RADIUS_KM = 5.0

COLUMNS = [
    "date_time",
    "latitude",
    "longitude",
    "potholes_detected",
    "confidence_scores",
    "distance_km",
]


# ---------------- DISTANCE CALCULATION ----------------

def distance_km(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two coordinates."""

    earth_radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 2 * earth_radius * math.asin(math.sqrt(a))


def valid_coordinates(latitude, longitude):
    return (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
    )


# ---------------- MODEL DOWNLOAD ----------------

def download_model():
    if MODEL_PATH.exists():
        return

    temporary_path = MODEL_PATH.with_suffix(".download")

    try:
        with requests.get(
            MODEL_URL, stream=True, timeout=(20, 180)
        ) as response:
            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)

        # Reject an HTML error page saved as a model file.
        with open(temporary_path, "rb") as file:
            if file.read(2) != b"PK":
                raise RuntimeError(
                    "Downloaded file is not a valid PyTorch "
                    "model archive. Check the model repository."
                )

        temporary_path.replace(MODEL_PATH)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@st.cache_resource
def load_model():
    download_model()
    return YOLO(str(MODEL_PATH))


# ---------------- REPORT STORAGE ----------------

def load_reports():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPORT_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    data = pd.read_csv(REPORT_FILE)

    for column in COLUMNS:
        if column not in data.columns:
            data[column] = None

    return data[COLUMNS]


def save_report(latitude, longitude, count, scores, distance):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    new_report = pd.DataFrame([{
        "date_time": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "latitude": latitude,
        "longitude": longitude,
        "potholes_detected": count,
        "confidence_scores": ", ".join(
            f"{score:.2f}" for score in scores
        ),
        "distance_km": round(distance, 3),
    }])

    existing = load_reports()
    updated = pd.concat(
        [existing, new_report], ignore_index=True
    )
    updated.to_csv(REPORT_FILE, index=False)


# ---------------- USER INTERFACE ----------------

st.set_page_config(
    page_title="NightRoad Guardian",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ NightRoad Guardian")
st.caption("AI-assisted pothole detection and 5 km reporting")

st.info(
    "Upload a road image, enter its GPS coordinates, "
    "and save detections inside your chosen 5 km radius."
)

st.sidebar.header("Monitoring area")

center_lat = st.sidebar.number_input(
    "Center latitude",
    min_value=-90.0,
    max_value=90.0,
    value=19.0,
    format="%.6f",
    help="Replace this example with your chosen center.",
)

center_lon = st.sidebar.number_input(
    "Center longitude",
    min_value=-180.0,
    max_value=180.0,
    value=73.0,
    format="%.6f",
    help="Replace this example with your chosen center.",
)

st.sidebar.write("Monitoring radius: 5 km")

st.sidebar.caption(
    "Example coordinates are not your actual GPS location."
)

st.subheader("1. Upload a road image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, PNG, or WEBP image",
    type=["jpg", "jpeg", "png", "webp"],
)

st.subheader("2. Enter the image location")

image_lat = st.number_input(
    "Image latitude",
    min_value=-90.0,
    max_value=90.0,
    value=19.0,
    format="%.6f",
)

image_lon = st.number_input(
    "Image longitude",
    min_value=-180.0,
    max_value=180.0,
    value=73.0,
    format="%.6f",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded road image")

    if st.button("Detect potholes", type="primary"):
        distance = distance_km(
            center_lat, center_lon, image_lat, image_lon
        )

        if distance > RADIUS_KM:
            st.error(
                f"This image location is {distance:.2f} km "
                "from the center, outside the 5 km radius."
            )
            st.stop()

        try:
            with st.spinner(
                "Loading AI model and analysing image..."
            ):
                model = load_model()
                result = model.predict(
                    source=image,
                    conf=0.25,
                    imgsz=640,
                    verbose=False,
                )[0]

            count = len(result.boxes)
            scores = (
                result.boxes.conf.cpu().tolist()
                if count > 0
                else []
            )

            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            st.subheader("Detection result")
            st.image(
                annotated_rgb,
                caption="AI predictions — inspect before reporting",
            )

            st.metric("Suspected potholes", count)
            st.write(f"Distance from center: {distance:.2f} km")

            if count:
                st.warning(
                    "AI detections are predictions, not confirmed "
                    "road defects. Review the marked image."
                )

                if st.button("Save pothole report"):
                    save_report(
                        image_lat,
                        image_lon,
                        count,
                        scores,
                        distance,
                    )
                    st.success("Report saved locally.")
            else:
                st.info(
                    "No potholes were detected in this image. "
                    "This does not prove the road is pothole-free."
                )

        except Exception as error:
            st.error(f"Detection failed: {error}")
            st.caption(
                "Check the internet connection, model download, "
                "and installed library versions."
            )


# ---------------- REPORT DASHBOARD ----------------

st.divider()
st.subheader("3. Reports within 5 km")

reports = load_reports()

if not reports.empty:
    reports["distance_km"] = reports.apply(
        lambda row: distance_km(
            center_lat,
            center_lon,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    nearby = reports[
        reports["distance_km"] <= RADIUS_KM
    ].copy()

    st.metric("Nearby saved reports", len(nearby))

    if not nearby.empty:
        st.dataframe(
            nearby.sort_values("distance_km"),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = nearby.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download nearby reports as CSV",
            data=csv_data,
            file_name="nightroad_nearby_reports.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved reports inside this radius yet.")
else:
    st.info("Your report database is empty. Test an image first.")

st.caption(
    "Prototype only. The app analyses uploaded images and uses "
    "coordinates entered by the user; it does not automatically "
    "track GPS or scan nearby roads."
)
