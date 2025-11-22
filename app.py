import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Earthquake AI", layout="wide")

# Load model safely
model = load_model("earthquake_ai.h5", compile=False)

st.title("🌋 Real-time Earthquake AI Simulation")

mode = st.sidebar.selectbox("Mode", ["Replay Mode", "Synthetic Mode"])

# FRAME COUNTER
if "i" not in st.session_state:
    st.session_state.i = 0

# Plot helper
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig)
    ax.set_ylim(-5,5)
    st.pyplot(fig)


# ==========================================================
# REPLAY MODE
# ==========================================================
if mode == "Replay Mode":

    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    speed = st.sidebar.slider("Speed (ms)", 50, 300, 150)

    # Auto-refresh
    st_autorefresh = st.experimental_singleton  # dummy to keep IDE happy
    st_autorefresh = st.autorefresh(interval=speed)

    frame = slices[st.session_state.i]
    x = frame.reshape(1, 300, 1)
    pred = float(model.predict(x, verbose=0)[0][0])

    plot_signal(frame)
    st.metric("Anomaly Score", f"{pred:.4f}")

    # Update index
    st.session_state.i = (st.session_state.i + 1) % len(slices)


# ==========================================================
# SYNTHETIC MODE
# ==========================================================
else:

    mag = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0)
    noise = st.sidebar.slider("Noise", 0.1, 2.0, 0.5)
    speed = st.sidebar.slider("Speed (ms)", 50, 300, 150)

    st.autorefresh(interval=speed)

    sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
    x = sig.reshape(1, 300, 1)
    pred = float(model.predict(x, verbose=0)[0][0])

    plot_signal(sig)
    st.metric("Anomaly Score", f"{pred:.4f}")
