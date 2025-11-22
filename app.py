import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Earthquake Early Warning AI System",
    layout="wide"
)

st.title("🌋 Earthquake Early Warning AI System")

# ---------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    anomaly = load_model("anomaly_model.h5", compile=False)
    magnitude = load_model("magnitude_model.h5", compile=False)
    return anomaly, magnitude

anomaly_model, magnitude_model = load_ai_models()


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def risk_level(anomaly, mag):
    if mag > 7 or anomaly > 0.75:
        return "🔴 YÜKSƏK RİSK"
    elif mag > 5 or anomaly > 0.45:
        return "🟡 ORTA RİSK"
    else:
        return "🟢 AŞAĞI RİSK"


def plot_signal(sig, return_fig=False):
    fig, ax = plt.subplots(figsize=(4.5, 2))

    ax.set_facecolor("#eef6fb")  # Ice blue background
    ax.plot(sig, color="#1f77b4", linewidth=1.5)

    ax.grid(True, color="#d0d7de", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_ylim(-5, 5)
    ax.tick_params(axis='both', labelsize=6, pad=2)
    ax.set_title("Seysmik Dalğa (son 2 saniyə)", fontsize=9)

    if return_fig:
        return fig
    else:
        st.pyplot(fig)


# -----------------------------------------------------------------------
# SYSTEM EXPLANATION — MARKDOWN TABLE (NO HTML, FULLY SAFE IN STREAMLIT)
# -----------------------------------------------------------------------
st.markdown("""
## 🧠 Sistem necə işləyir?

| Komponent | İzahetmə |
|----------|-----------|
| **📈 Anomaly Score** | Dalğadakı qeyri-normal dəyişikliklərin gücünü ölçür. <br>• 0.0–0.3 → Normal <br>• 0.3–0.7 → Orta <br>• 0.7+ → Güclü zəlzələ əlaməti |
| **🌋 Magnitude Proqnozu** | Dalğanın formasına əsasən AI-nin təxmini magnitude qiyməti (3–8). Real magnitude deyil — siqnaldan çıxan AI proqnozudur. |
| **🔊 Noise (Səs-küy)** | Siqnala əlavə edilən təsadüfi dəyişikliklərdir. <br>• Noise ↑ → xaotik dalğa <br>• Noise ↓ → təmiz dalğa |
| **📡 Real-Time Simulyasiya** | Parametrlər dəyişdikcə dalğa və AI nəticələri real vaxtda yenilənir. |
| **🖼 Statik Göstərici** | Sabit dalğa göstərilir və AI nəticələri dəyişmir. |
""", unsafe_allow_html=False)

st.divider()

# ---------------------------------------------------------------
# PRESET FIX — ALWAYS RESPONSIVE
# ---------------------------------------------------------------
if "preset" not in st.session_state:
    st.session_state["preset"] = None

def set_preset(p):
    st.session_state["preset"] = p


# ---------------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------------
mode = st.sidebar.radio("Rejim seç:", ["Real-time Simulyasiya", "Statik Göstərici"])


# ---------------------------------------------------------------
# REAL-TIME MODE
# ---------------------------------------------------------------
if mode == "Real-time Simulyasiya":

    # -----------------------------------------
    # ⭐ PRESET BUTTONS NOW AT THE TOP HERE ⭐
    # -----------------------------------------
    st.subheader("🧪 AI-ni sınağa çək")

    colW, colM, colS = st.columns(3)

    colW.button("🟢 Weak Quake", on_click=set_preset, args=("weak",))
    colM.button("🟡 Medium Quake", on_click=set_preset, args=("medium",))
    colS.button("🔴 Strong Quake", on_click=set_preset, args=("strong",))

    st.write("")  # spacing

    st.header("⚙️ Parametrlər")

    mag_input = st.slider("Magnitude", 3.0, 8.0, 5.0)
    noise_input = st.slider("Səs-küy (Noise)", 0.1, 2.0, 0.5)

    # Apply preset BEFORE generating predictions
    if st.session_state["preset"] == "weak":
        st.session_state["sig"] = generate_signal(4.0, 0.2)
    elif st.session_state["preset"] == "medium":
        st.session_state["sig"] = generate_signal(5.5, 0.4)
    elif st.session_state["preset"] == "strong":
        st.session_state["sig"] = generate_signal(7.0, 0.7)
    else:
        if "sig" not in st.session_state:
            st.session_state["sig"] = generate_signal(5.0, 0.5)

    if st.button("Yeni dalğa yarat"):
        st.session_state["preset"] = None
        st.session_state["sig"] = generate_signal(mag_input, noise_input)

    sig = st.session_state["sig"]

    X = sig.reshape(1, 300, 1)
    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📈 Anomaly Score", f"{anomaly:.3f}")
    with col2:
        st.metric("🌋 Magnitude", f"{predicted_mag:.2f}")
    with col3:
        st.metric("⚠️ Risk", risk)

    plot_signal(sig)
    st.caption("Bu qrafik son 2 saniyəlik seysmik dalğanı göstərir. AI bu siqnaldan anomaliya və magnitude təxminini çıxarır.")


# ---------------------------------------------------------------
# STATIC MODE
# ---------------------------------------------------------------
else:
    st.header("📡 Statik Nümunə Dalğa")

    sig = generate_signal(5.0, 0.5)
    X = sig.reshape(1, 300, 1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📈 Anomaly Score", f"{anomaly:.3f}")
    with col2:
        st.metric("🌋 Magnitude", f"{predicted_mag:.2f}")
    with col3:
        st.metric("⚠️ Risk", risk)

    plot_signal(sig)
    st.caption("Bu qrafik təlim məqsədlidir. Dalğa sabitdir və dəyişmir.")


# ---------------------------------------------------------------
# NOISE VISUALIZATION
# ---------------------------------------------------------------
st.divider()
st.header("🔍 Noise təsirini vizual müqayisə et")

colA, colB = st.columns(2)

noise_test = st.slider("Noise dəyərini seç (vizual müqayisə üçün):", 0.1, 2.0, 0.5, 0.1)

with colA:
    st.write("**Təmiz Dalğa (Noise = 0.1)**")
    clean = generate_signal(5.0, 0.1)
    plot_signal(clean)

with colB:
    st.write(f"**Səs-küylü Dalğa (Noise = {noise_test})**")
    noisy = generate_signal(5.0, noise_test)
    plot_signal(noisy)


# ---------------------------------------------------------------
# SEISMOGRAPH REPLAY
# ---------------------------------------------------------------
st.divider()
st.header("🎞 Seismograph Replay (5 saniyə)")

if st.button("▶ Başlat Replay"):
    placeholder = st.empty()

    for i in range(25):
        sig = generate_signal(5.0, 0.5)
        fig = plot_signal(sig, return_fig=True)
        placeholder.pyplot(fig)
        time.sleep(0.2)
