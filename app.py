import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Earthquake AI", layout="wide")

# Load H5 model
model = load_model("earthquake_ai.h5", compile=False)

st.title("🌋 Real-time Earthquake AI Simulation")

# State setup
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0

mode = st.sidebar.selectbox("Mode", ["Replay Mode", "Synthetic Mode"])

# ANY slider change stops the stream
def stop_stream():
    st.session_state.streaming = False

# Helper for plotting
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(sig)
    ax.set_ylim(-5, 5)
    st.pyplot(fig)


# ================================================
# REPLAY MODE
# ================================================
if mode == "Replay Mode":
    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    speed = st.sidebar.slider("Speed (ms)", 50, 300, 150, on_change=stop_stream)

    if st.button("▶ Start Replay"):
        st.session_state.streaming = True

    if st.button("⏹ Stop"):
        st.session_state.streaming = False

    # STREAMING
    if st.session_state.streaming:

        frame = slices[st.session_state.frame_index]
        x = frame.reshape(1, 300, 1)
        pred = float(model.predict(x, verbose=0)[0][0])

        plot_signal(frame)
        st.metric("Anomaly Score", f"{pred:.4f}")

        # increment safely
        st.session_state.frame_index = (st.session_state.frame_index + 1) % len(slices)

        # auto-refresh
        st.experimental_set_query_params(refresh="1") 
        st.sleep(speed / 1000)

    else:
        st.write("Press ▶ Start to begin.")


# ================================================
# SYNTHETIC MODE
# ================================================
else:
    mag = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0, on_change=stop_stream)
    noise = st.sidebar.slider("Noise", 0.1, 2.0, 0.5, on_change=stop_stream)
    speed = st.sidebar.slider("Speed (ms)", 50, 300, 150, on_change=stop_stream)

    if st.button("▶ Start Synthetic Stream"):
        st.session_state.streaming = True

    if st.button("⏹ Stop"):
        st.session_state.streaming = False

    if st.session_state.streaming:

        sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
        x = sig.reshape(1, 300, 1)
        pred = float(model.predict(x, verbose=0)[0][0])

        plot_signal(sig)
        st.metric("Anomaly Score", f"{pred:.4f}")

        st.experimental_set_query_params(refresh="1")
        st.sleep(speed / 1000)

    else:
        st.write("Press ▶ Start to begin.")
