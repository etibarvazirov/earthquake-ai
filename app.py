import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Earthquake Early Warning AI System", layout="wide")


# ---------------------------------------------------------
# GLOBAL CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<div class='info-card'>
    <div class='info-title'>🧠 Sistem necə işləyir?</div>
    <div class='info-desc'>
        <b>1️⃣ Anomaly Score</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            Dalğanın strukturundakı qeyri-adi dəyişikliklərin gücünü ölçür.<br>
            0.0 → normal<br>
            0.3 → orta<br>
            0.7+ → güclü zəlzələ əlaməti
            </span>
        </span>
        <br><br>

        <b>2️⃣ Magnitude Proqnozu</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            AI dalğadan magnitude proqnozu verir.
            </span>
        </span>
        <br><br>

        <b>3️⃣ Zəlzələ Riski</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            🟢 Aşağı risk<br>
            🟡 Orta risk<br>
            🔴 Yüksək risk
            </span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🌋 Earthquake Early Warning AI System")


# ---------------------------------------------------------
# MAIN INFO CARD
# ---------------------------------------------------------
st.markdown("""
<div class='info-card'>
    <div class='info-title'>🧠 Sistem necə işləyir?</div>
    <div class='info-desc'>
        Bu AI sistemi seysmik dalğaları analiz edib <b>üç əsas göstərici</b> çıxarır.<br><br>

        <b>1️⃣ Anomaly Score</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            Dalğanın strukturundakı qeyri-adi dəyişikliklərin gücünü ölçür.<br>
            0.0 → normal dalğa<br>
            0.3 → orta anomaliya<br>
            0.7+ → güclü zəlzələ əlaməti
            </span>
        </span>
        <br><br>

        <b>2️⃣ Magnitude Proqnozu</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            AI dalğanın gücünə baxaraq təxmini zəlzələ magnitude-ni proqnozlaşdırır (3.0–8.0).
            </span>
        </span>
        <br><br>

        <b>3️⃣ Zəlzələ Riski</b>
        <span class='tooltip'>ℹ️
            <span class='tooltiptext'>
            Anomaly Score və Magnitude birlikdə analiz edilərək yekun risk çıxarılır.<br>
            🟢 Aşağı risk<br>
            🟡 Orta risk<br>
            🔴 Yüksək risk
            </span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD AI MODELS
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
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(sig, color="black")
    ax.set_ylim(-5,5)
    ax.set_title("Seysmik dalğa (son 2 saniyə)")
    st.pyplot(fig)


# ---------------------------------------------------------
# SIDEBAR: INFO BOX + THEME + MODE
# ---------------------------------------------------------
with st.sidebar.expander("ℹ️ Bu panel nə edir?"):
    st.write("""
    Bu simulyasiya AI tərəfindən yaradılmış seysmik dalğaları analiz edir.

    - Yeni dalğa → AI həm Anomaly Score, həm də Magnitude proqnozu çıxarır  
    - Risk → hər iki göstəricinin kombinasiyası  
    """)

theme_choice = st.sidebar.radio("Tema:", ["Light", "Dark"])

if theme_choice == "Dark":
    st.markdown("<script>document.body.classList.add('dark-mode');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('dark-mode');</script>", unsafe_allow_html=True)


mode = st.sidebar.radio("Rejim seç:", ["Real-time Simulyasiya", "Statik göstərici"])


# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if mode == "Real-time Simulyasiya":

    st.sidebar.subheader("Parametrlər")
    mag_input = st.sidebar.slider("Magnitude", 3.0, 8.0, 5.0)
    noise_input = st.sidebar.slider("Səs-küy", 0.1, 2.0, 0.5)

    if st.button("Yeni seysmik dalğa yarat"):
        sig = generate_signal(mag=mag_input, noise_level=noise_input)
        st.session_state["sig"] = sig

    if "sig" not in st.session_state:
        st.session_state["sig"] = generate_signal(5.0, 0.5)

    sig = st.session_state["sig"]

    X = sig.reshape(1,300,1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    # KPI CARDS
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Anomaly Score</div>
            <div class='kpi-value'>{anomaly:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Magnitude Proqnozu</div>
            <div class='kpi-value'>{predicted_mag:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Risk Səviyyəsi</div>
            <div class='kpi-value'>{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    plot_signal(sig)

    st.caption("Bu qrafik son 2 saniyəlik seysmik dalğanı göstərir. AI bu siqnaldan anomaliya və magnitude təxminini çıxarır.")


else:
    sig = generate_signal(5.0, 0.5)
    X = sig.reshape(1,300,1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Anomaly Score</div>
            <div class='kpi-value'>{anomaly:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Magnitude Proqnozu</div>
            <div class='kpi-value'>{predicted_mag:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Risk Səviyyəsi</div>
            <div class='kpi-value'>{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    plot_signal(sig)

    st.caption("Bu qrafik son 2 saniyəlik seysmik dalğanı göstərir. AI bu siqnaldan anomaliya və magnitude təxminini çıxarır.")
