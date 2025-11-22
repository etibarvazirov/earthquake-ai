import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Earthquake Early Warning AI System", layout="wide")

# -------------------------------------------------------------------
# THEME SELECTION FIRST (JS YOXDUR, TAM DÜZGÜN İŞLƏYİR)
# -------------------------------------------------------------------
theme_choice = st.sidebar.radio("Tema:", ["Light", "Dark"])

if theme_choice == "Dark":
    card_bg = "#2a2a2a"
    text_color = "#f2f2f2"
    border_color = "#5AB9EA"
else:
    card_bg = "#f5f5f7"
    text_color = "#1a1a1a"
    border_color = "#4B9CD3"

# Apply dynamic CSS
st.markdown(f"""
<style>
.info-card {{
    background-color: {card_bg};
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid {border_color};
    margin-bottom: 20px;
    color: {text_color};
}}

.info-title {{
    font-size: 22px;
    font-weight: bold;
}}

.info-desc {{
    font-size: 16px;
    margin-left: 10px;
    line-height: 1.5;
}}

.kpi-card {{
    background-color: {card_bg};
    padding: 15px;
    border-radius: 10px;
    border: 2px solid {border_color};
    text-align: center;
    margin-bottom: 10px;
    color: {text_color};
}}

.kpi-value {{
    font-size: 26px;
    font-weight: bold;
}}

.kpi-title {{
    font-size: 16px;
    opacity: 0.9;
    color: {text_color};
}}

.tooltip {{
    position: relative;
    display: inline-block;
    cursor: help;
    color: #4BA3FF;
}}

.tooltip .tooltiptext {{
    visibility: hidden;
    width: 260px;
    background-color: {border_color};
    color: white;
    text-align: left;
    border-radius: 6px;
    padding: 10px;
    position: absolute;
    z-index: 10;
    bottom: 125%;
    left: 50%;
    margin-left: -130px;
    opacity: 0;
    transition: opacity 0.4s;
}}

.tooltip:hover .tooltiptext {{
    visibility: visible;
    opacity: 1;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TITLE
# -------------------------------------------------------------------
st.title("🌋 Earthquake Early Warning AI System")

# -------------------------------------------------------------------
# BIG INFO CARD
# -------------------------------------------------------------------
st.markdown("""
<div class='info-card'>
    <div class='info-title'>🧠 Sistem necə işləyir?</div>

    <div class='info-desc'>
        <b>1️⃣ Anomaly Score</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
                Dalğadakı qeyri-adi dəyişikliklərin gücünü ölçür.<br>
                0.0 → normal<br>
                0.3 → orta<br>
                0.7+ → güclü zəlzələ əlaməti
            </span>
        </span>
        <br><br>

        <b>2️⃣ Magnitude Proqnozu</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
                AI dalğa formasına baxaraq təxmini magnitude proqnozu verir (3.0 – 8.0).
            </span>
        </span>
        <br><br>

        <b>3️⃣ Zəlzələ Riski</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
                Anomaly Score + Magnitude birlikdə analiz edilir:<br>
                🟢 Aşağı risk<br>
                🟡 Orta risk<br>
                🔴 Yüksək risk
            </span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD MODELS (SAFE CACHE)
# -------------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    anomaly = load_model("anomaly_model.h5", compile=False)
    magnitude = load_model("magnitude_model.h5", compile=False)
    return anomaly, magnitude

anomaly_model, magnitude_model = load_ai_models()

# -------------------------------------------------------------------
# RISK ENGINE
# -------------------------------------------------------------------
def risk_level(anomaly, mag):
    if mag > 7 or anomaly > 0.75:
        return "🔴 YÜKSƏK RİSK"
    elif mag > 5 or anomaly > 0.45:
        return "🟡 ORTA RİSK"
    else:
        return "🟢 AŞAĞI RİSK"

# -------------------------------------------------------------------
# PLOTTER
# -------------------------------------------------------------------
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig, color="black")
    ax.set_ylim(-5,5)
    ax.set_title("Seysmik dalğa (son 2 saniyə)")
    st.pyplot(fig)

# -------------------------------------------------------------------
# SIDEBAR INFO
# -------------------------------------------------------------------
with st.sidebar.expander("ℹ️ Bu panel nə edir?"):
    st.write("""
    Sistem real-time simulyasiya edilmiş seysmik siqnalları AI modelləri ilə təhlil edir.
    - Anomaliya → qeyri-adi dəyişikliklərin gücü  
    - Magnitude → dalğanın gücü  
    - Risk → hər ikisinin kombinasiyası  
    """)

mode = st.sidebar.radio("Rejim seç:", ["Real-time Simulyasiya", "Statik göstərici"])

# -------------------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------------------
if mode == "Real-time Simulyasiya":

    st.sidebar.subheader("Parametrlər")
    mag_input = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0)
    noise_input = st.sidebar.slider("Səs-küy", 0.1, 2.0, 0.5)

    if st.button("Yeni dalğa yarat"):
        st.session_state["sig"] = generate_signal(mag_input, noise_input)

    if "sig" not in st.session_state:
        st.session_state["sig"] = generate_signal(5.0, 0.5)

    sig = st.session_state["sig"]

    X = sig.reshape(1, 300, 1)
    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Anomaly Score</div><div class='kpi-value'>{anomaly:.3f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Magnitude</div><div class='kpi-value'>{predicted_mag:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Risk</div><div class='kpi-value'>{risk}</div></div>", unsafe_allow_html=True)

    plot_signal(sig)
    st.caption("Bu qrafik son 2 saniyəlik seysmik dalğanı göstərir. AI bu siqnaldan anomaliya və magnitude təxminini çıxarır.")

else:
    sig = generate_signal(5.0, 0.5)
    X = sig.reshape(1,300,1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Anomaly Score</div><div class='kpi-value'>{anomaly:.3f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Magnitude</div><div class='kpi-value'>{predicted_mag:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Risk</div><div class='kpi-value'>{risk}</div></div>", unsafe_allow_html=True)

    plot_signal(sig)
    st.caption("Bu qrafik son 2 saniyəlik seysmik dalğanı göstərir. AI bu siqnaldan anomaliya və magnitude təxminini çıxarır.")
