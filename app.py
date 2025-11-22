import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Earthquake Early Warning AI System", layout="wide")

st.title("🌋 Earthquake Early Warning AI System")


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_ai_models():
    anomaly = load_model("anomaly_model.h5", compile=False)
    magnitude = load_model("magnitude_model.h5", compile=False)
    return anomaly, magnitude

anomaly_model, magnitude_model = load_ai_models()


# ---------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------
def risk_level(anomaly, mag):
    if mag > 7 or anomaly > 0.75:
        return "🔴 YÜKSƏK RİSK"
    elif mag > 5 or anomaly > 0.45:
        return "🟡 ORTA RİSK"
    else:
        return "🟢 AŞAĞI RİSK"


# ---------------------------------------------------------
# PLOTTER
# ---------------------------------------------------------
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(sig, color="black")
    ax.set_ylim(-5, 5)
    ax.set_title("Seysmik dalğa (son 2 saniyə)")
    st.pyplot(fig)


# ---------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------
mode = st.sidebar.radio("Rejim seç:", ["Real-time Simulyasiya", "Statik göstərici"])


# ---------------------------------------------------------
# REAL-TIME SIMULATION
# ---------------------------------------------------------
if mode == "Real-time Simulyasiya":

    st.sidebar.subheader("Parametrlər")
    mag_input = st.sidebar.slider("Magnitude (təxmini güc)", 3.0, 8.0, 5.0)
    noise_input = st.sidebar.slider("Səs-küy səviyyəsi", 0.1, 2.0, 0.5)

    if st.button("Yeni siqnal yarat"):
        sig = generate_signal(mag=mag_input, noise_level=noise_input)
        st.session_state["last_sig"] = sig

    # İlk açılışda siqnal yarat
    if "last_sig" not in st.session_state:
        st.session_state["last_sig"] = generate_signal(mag=5.0, noise_level=0.5)

    sig = st.session_state["last_sig"]

    # Model input
    X = sig.reshape(1, 300, 1)

    # Predictions
    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])

    # RISK estimation
    risk = risk_level(anomaly, predicted_mag)

    # Show results
    st.subheader(f"Zəlzələ riski: {risk}")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Anomaly Score", f"{anomaly:.3f}")

    with col2:
        st.metric("AI Magnitude Proqnozu", f"{predicted_mag:.2f}")

    plot_signal(sig)

    st.info(
        "Bu panel AI tərəfindən yaradılmış seysmik dalğaları analiz edir.\n"
        "Model dalğanın strukturunu təhlil edərək həm **anomaliya dərəcəsini**, "
        "həm də **təxmini magnitude-ni** proqnozlaşdırır."
    )


# ---------------------------------------------------------
# STATIC MODE
# ---------------------------------------------------------
else:
    st.write("Bu rejimdə model yalnız göstərilən siqnala əsasən nəticə verir.")
    sig = generate_signal(5.0, 0.5)
    plot_signal(sig)

    X = sig.reshape(1, 300, 1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    st.subheader(f"Zəlzələ riski: {risk}")
    st.metric("Anomaly Score", f"{anomaly:.3f}")
    st.metric("Magnitude Proqnozu", f"{predicted_mag:.2f}")
