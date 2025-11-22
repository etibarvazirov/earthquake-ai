import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Real-time Earthquake AI Platform", layout="wide")

model = load_model("earthquake_ai.h5")

st.title("🌋 Real-time Earthquake AI Simulation Platform")

mode = st.sidebar.selectbox("Mode seç:", ["Replay Mode", "Synthetic Mode"])


# --- PLOT FUNKSIYASI ---
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig)
    ax.set_ylim(-5,5)
    ax.set_title("Live Seismic Signal")
    return fig


# =======================================================
# MODE 1 — REPLAY MODE (fixed: no rerun)
# =======================================================
if mode == "Replay Mode":
    st.header("📡 Real Earthquake Replay Mode")
    slices = np.load("real_slices.npy")

    speed = st.sidebar.slider("Stream Speed (sec)", 0.01, 0.3, 0.05)

    # Containers
    plot_container = st.empty()
    score_container = st.empty()

    for frame in slices:
        frame_r = frame.reshape(1, -1, 1)
        pred = model.predict(frame_r)[0][0]

        with plot_container:
            st.pyplot(plot_signal(frame))

        with score_container:
            st.metric("Anomaly Score", f"{pred:.4f}")

        time.sleep(speed)


# =======================================================
# MODE 2 — SYNTHETIC MODE (fixed: no rerun)
# =======================================================
else:
    st.header("🧪 Synthetic Earthquake Generator")

    mag = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0)
    noise = st.sidebar.slider("Noise Level", 0.1, 2.0, 0.5)

    speed = st.sidebar.slider("Stream Speed (sec)", 0.01, 0.3, 0.05)

    # Containers
    plot_container = st.empty()
    score_container = st.empty()

    while True:
        sig = generate_signal(mag=mag, noise_level=noise)
        frame_r = sig.reshape(1, -1, 1)
        pred = model.predict(frame_r)[0][0]

        with plot_container:
            st.pyplot(plot_signal(sig))

        with score_container:
            st.metric("Anomaly Score", f"{pred:.4f}")

        time.sleep(speed)
