import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Earthquake AI", layout="wide")

model = load_model("earthquake_ai.h5", compile=False)

st.title("🌋 Earthquake AI — Step Mode (Stable)")

mode = st.sidebar.selectbox("Mode", ["Replay", "Synthetic"])

def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig)
    ax.set_ylim(-5,5)
    st.pyplot(fig)


if "idx" not in st.session_state:
    st.session_state.idx = 0


# ------------------------------
# REPLAY MODE
# ------------------------------
if mode == "Replay":
    st.header("Real Seismic Replay")

    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    if st.button("Next frame"):
        st.session_state.idx += 1

    frame = slices[st.session_state.idx % len(slices)]
    pred = float(model.predict(frame.reshape(1,300,1), verbose=0)[0][0])

    plot_signal(frame)
    st.metric("Anomaly Score", f"{pred:.4f}")


# ------------------------------
# SYNTHETIC MODE
# ------------------------------
else:
    st.header("Synthetic Generator")

    mag = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0)
    noise = st.sidebar.slider("Noise", 0.1, 2.0, 0.5)

    if st.button("Generate new"):
        st.session_state.idx += 1

    sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
    pred = float(model.predict(sig.reshape(1,300,1), verbose=0)[0][0])

    plot_signal(sig)
    st.metric("Anomaly Score", f"{pred:.4f}")
