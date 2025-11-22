import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal
import tensorflow as tf

st.set_page_config(page_title="Earthquake AI", layout="wide")

# Load model safely
model = tf.keras.models.load_model("earthquake_ai", compile=False)

st.title("🌋 Real-time Earthquake AI Simulation")

mode = st.sidebar.selectbox("Mode seç:", ["Replay Mode", "Synthetic Mode"])

# State for streaming
if "running" not in st.session_state:
    st.session_state.running = False

# Stop stream when sliders change
def stop_stream():
    st.session_state.running = False

st.sidebar.write("### Controls")


def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig)
    ax.set_ylim(-5,5)
    return fig


# =====================================================
# REPLAY MODE
# =====================================================
if mode == "Replay Mode":

    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    speed = st.sidebar.slider("Speed", 0.01, 0.3, 0.05, on_change=stop_stream)

    start_btn = st.button("▶ Start Replay")
    stop_btn  = st.button("⏹ Stop")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    plot_box = st.empty()
    text_box = st.empty()

    if st.session_state.running:
        # One frame per rerun
        idx = st.session_state.get("frame_idx", 0)

        frame = slices[idx]
        frame_r = frame.reshape(1, 300, 1)

        pred = float(model.predict(frame_r, verbose=0)[0][0])

        plot_box.pyplot(plot_signal(frame))
        text_box.metric("Anomaly Score", f"{pred:.4f}")

        # update index
        st.session_state.frame_idx = (idx + 1) % len(slices)

        time.sleep(speed)
        st.experimental_rerun()

    else:
        st.session_state.frame_idx = 0



# =====================================================
# SYNTHETIC MODE
# =====================================================
else:
    mag = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0, on_change=stop_stream)
    noise = st.sidebar.slider("Noise", 0.1, 2.0, 0.5, on_change=stop_stream)
    speed = st.sidebar.slider("Speed", 0.01, 0.3, 0.05, on_change=stop_stream)

    start_btn = st.button("▶ Start Synthetic Stream")
    stop_btn  = st.button("⏹ Stop")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    plot_box = st.empty()
    text_box = st.empty()

    if st.session_state.running:
        sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
        frame_r = sig.reshape(1, 300, 1)

        pred = float(model.predict(frame_r, verbose=0)[0][0])

        plot_box.pyplot(plot_signal(sig))
        text_box.metric("Anomaly Score", f"{pred:.4f}")

        time.sleep(speed)
        st.experimental_rerun()
