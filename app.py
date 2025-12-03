import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

import streamlit as st

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Earthquake Early Warning AI System",
    layout="wide"
)

# ===============================================================
# 🌙 DARK MODE TOGGLE
# ===============================================================
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

if dark_mode:
    BG_COLOR = "#1e1e1e"
    TEXT_COLOR = "#f2f2f2"
    CARD_BG = "#2a2a2a"
    BORDER = "#4B9CD3"
    BANNER_GRAD = "linear-gradient(90deg, #003566, #001d3d)"
else:
    BG_COLOR = "#ffffff"
    TEXT_COLOR = "#1a1a1a"
    CARD_BG = "#f5f5f7"
    BORDER = "#4B9CD3"
    BANNER_GRAD = "linear-gradient(90deg, #4fa3f7, #005fbb)"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    .main-title {{
        background: {BANNER_GRAD};
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
        margin-bottom: 25px;
    }}
    .subtext {{
        font-size: 16px;
        opacity: 0.9;
        margin-top: -10px;
    }}
    .info-box {{
        background-color: {CARD_BG};
        color: {TEXT_COLOR};
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid {BORDER};
        box-shadow: 0px 2px 5px rgba(0,0,0,0.15);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# TOP BANNER
st.markdown("""
<div class="main-title">
🌋 Earthquake Early Warning AI System
<div class="subtext">AI ilə real-time seysmik analiz və risk proqnozlaşdırılması</div>
</div>
""", unsafe_allow_html=True)


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
# RISK + PLOT
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

    ax.set_facecolor("#dbe7f3" if not dark_mode else "#2b2b2b")
    ax.plot(sig, color="#1f77b4", linewidth=1.5)

    ax.grid(
        True, color="#c3ccd5" if not dark_mode else "#555",
        linestyle="--", linewidth=0.5, alpha=0.6
    )
    ax.set_ylim(-5, 5)
    ax.tick_params(axis='both', labelsize=6, pad=2)
    ax.set_title("Seysmik Dalğa (son 2 saniyə)", fontsize=9, color=TEXT_COLOR)

    if return_fig:
        return fig
    else:
        st.pyplot(fig)


# ---------------------------------------------------------------
# SYSTEM TABLE (inside styled card)
# ---------------------------------------------------------------
st.markdown(f"""
<div class="info-box">
<h3>🧠 Sistem necə işləyir?</h3>

<div style="font-size:16px">
<table style="width:100%; border-collapse: collapse;">
<tr><td><b>📈 Anomaly Score</b></td><td>Dalğadakı qeyri-normal dəyişikliklərin gücünü göstərir. 0.0–0.3 → Normal, 0.3–0.7 → Orta, 0.7+ → Güclü siqnal.</td></tr>
<tr><td><b>🌋 Magnitude Proqnozu</b></td><td>Dalğanın formasına əsasən AI tərəfindən hesablanan təxmini qiymət (3–8).</td></tr>
<tr><td><b>🔊 Noise</b></td><td>Dalğaya səs-küy əlavə edir. Noise ↑ → xaotik dalğa.</td></tr>
<tr><td><b>📡 Real-Time Simulyasiya</b></td><td>Parametrlər dəyişdikcə bütün nəticələr dərhal yenilənir.</td></tr>
<tr><td><b>🖼 Statik Göstərici</b></td><td>Sabit dalğa göstərir. Modelin davranışını izah etmək üçündür.</td></tr>
</table>
</div>

</div>
""", unsafe_allow_html=True)

st.divider()


# ---------------------------------------------------------------
# PRESET LOGIC
# ---------------------------------------------------------------
if "preset" not in st.session_state:
    st.session_state["preset"] = None

def set_preset(p):
    st.session_state["preset"] = p


# ---------------------------------------------------------------
# MODE SELECT
# ---------------------------------------------------------------
mode = st.sidebar.radio("Rejim seç:", ["Real-time Simulyasiya", "Statik Göstərici"])


# ---------------------------------------------------------------
# REAL TIME MODE
# ---------------------------------------------------------------
if mode == "Real-time Simulyasiya":

    st.subheader("🧪 AI-ni sınağa çək")

    colW, colM, colS = st.columns(3)
    colW.button("🟢 Weak Quake", on_click=set_preset, args=("weak",))
    colM.button("🟡 Medium Quake", on_click=set_preset, args=("medium",))
    colS.button("🔴 Strong Quake", on_click=set_preset, args=("strong",))

    st.header("⚙️ Parametrlər")

    mag_input = st.slider("Magnitude", 3.0, 8.0, 5.0)
    noise_input = st.slider("Səs-küy (Noise)", 0.1, 2.0, 0.5)

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

    col1.metric("📈 Anomaly Score", f"{anomaly:.3f}")
    col2.metric("🌋 Magnitude", f"{predicted_mag:.2f}")
    col3.metric("⚠️ Risk", risk)

    plot_signal(sig)


# ---------------------------------------------------------------
# STATIC MODE
# ---------------------------------------------------------------
else:
    st.header("📡 Statik Nümunə Dalğa")
    st.info("Bu rejim sabit dalğa yaradır və AI nəticələri dəyişmir. "
            "Modelin davranışını izah etmək üçün istifadə olunur.")

    sig = generate_signal(5.0, 0.5)
    X = sig.reshape(1, 300, 1)

    anomaly = float(anomaly_model.predict(X, verbose=0)[0][0])
    predicted_mag = float(magnitude_model.predict(X, verbose=0)[0][0])
    risk = risk_level(anomaly, predicted_mag)

    col1, col2, col3 = st.columns(3)

    col1.metric("📈 Anomaly Score", f"{anomaly:.3f}")
    col2.metric("🌋 Magnitude", f"{predicted_mag:.2f}")
    col3.metric("⚠️ Risk", risk)

    plot_signal(sig)


# ---------------------------------------------------------------
# NOISE VISUALIZATION
# ---------------------------------------------------------------
st.divider()
st.header("🔍 Noise təsirini vizual müqayisə et")

colA, colB = st.columns(2)
noise_test = st.slider("Noise dəyəri:", 0.1, 2.0, 0.5, 0.1)

with colA:
    st.write("**Təmiz Dalğa (Noise = 0.1)**")
    plot_signal(generate_signal(5.0, 0.1))

with colB:
    st.write(f"**Səs-küylü Dalğa (Noise = {noise_test})**")
    plot_signal(generate_signal(5.0, noise_test))


# ---------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------
st.divider()
st.header("🎞 Real-time Seismograph Replay")

if st.button("▶ Başlat Replay"):
    placeholder = st.empty()

    for i in range(25):
        sig = generate_signal(5.0, 0.5)
        fig = plot_signal(sig, return_fig=True)
        placeholder.pyplot(fig)
        time.sleep(0.2)
