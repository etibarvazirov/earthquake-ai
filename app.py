import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Earthquake AI", layout="wide")

model = load_model("earthquake_ai.h5", compile=False)

st.title("🌋 Earthquake Early Warning AI System")

# Counter for frames
if "i" not in st.session_state:
    st.session_state.i = 0


# -------- Risk conversion --------
def risk_level(score):
    if score > 0.65:
        return "🔴 YÜKSƏK RİSK"
    elif score > 0.35:
        return "🟡 ORTA RİSK"
    else:
        return "🟢 AŞAĞI RİSK"


# -------- Plot helper --------
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(sig, color="black")
    ax.set_ylim(-5,5)
    ax.set_title("Son 2 saniyəlik seysmik dalğa")
    st.pyplot(fig)


# -------- Mode selection --------
mode = st.sidebar.radio("Rejim seç", ["Replay (real data)", "Synthetic (AI simulyasiya)"])


# -------- Replay Mode --------
if mode == "Replay (real data)":

    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    # auto-refresh every 150ms
    st.autorefresh(interval=150)

    frame = slices[st.session_state.i % len(slices)]
    x = frame.reshape(1, 300, 1)
    score = float(model.predict(x, verbose=0)[0][0])

    # RISK PANEL
    st.subheader(f"Zəlzələ riski: {risk_level(score)}")

    # GRAPH
    plot_signal(frame)

    # INFO
    st.info(
        "Bu panel canlı seysmik dalğaları analiz edir.\n"
        "AI modeli dalğadakı qeyri-adi dəyişiklikləri taparaq risk səviyyəsini proqnozlaşdırır.\n"
        "Risk səviyyəsi anomaliya skoruna əsasən hesablanır."
    )

    st.session_state.i += 1


# -------- Synthetic Mode --------
else:
    mag = st.sidebar.slider("Süni zəlzələ gücü (Magnitude)", 3.0, 8.0, 5.0)
    noise = st.sidebar.slider("Səs-küy", 0.1, 2.0, 0.5)

    st.autorefresh(interval=150)

    sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
    x = sig.reshape(1, 300, 1)
    score = float(model.predict(x, verbose=0)[0][0])

    # RISK PANEL
    st.subheader(f"Zəlzələ riski: {risk_level(score)}")

    # GRAPH
    plot_signal(sig)

    st.info(
        "Bu simulyasiya rejimində AI-ya süni seysmik dalğa verilir.\n"
        "Dalğanın gücü (Magnitude) artırıldıqca risk səviyyəsi yüksəlir."
    )
