import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# Custom background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('background.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp > header, .stApp > footer {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and info
st.title("🔥 Fire Type Classification")
st.markdown("Predict fire type based on MODIS satellite readings.")

# Input fields with validation
brightness = st.number_input("Brightness", min_value=100.0, max_value=500.0, value=300.0)
bright_t31 = st.number_input("Brightness T31", min_value=200.0, max_value=350.0, value=290.0)
frp = st.number_input("Fire Radiative Power (FRP)", min_value=0.0, max_value=300.0, value=15.0)
scan = st.number_input("Scan", min_value=0.1, max_value=10.0, value=1.0)
track = st.number_input("Track", min_value=0.1, max_value=10.0, value=1.0)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Predict and display
if st.button("Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, f"Unknown (class index: {prediction})")
    st.success(f"**Predicted Fire Type:** {result}")
