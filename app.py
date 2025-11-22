import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from quake_generator import generate_signal

st.set_page_config(page_title="Earthquake AI", layout="wide")

model = load_model("earthquake_ai.h5", compile=False)

st.title("🌋 Earthquake Early Warning AI System")

# State for frame counter
if "i" not in st.session_state:
    st.session_state.i = 0

# Risk label
def risk_label(score):
    if score > 0.65:
        return "🔴 Yüksək risk (zəlzələ ehtimalı artıb)"
    elif score > 0.35:
        return "🟡 Orta risk (dalğada narahatlıq var)"
    else:
        return "🟢 Aşağı risk (hər şey normaldır)"

# Plot function
def plot_signal(sig):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(sig, color="black")
    ax.set_ylim(-5,5)
    ax.set_title("Seysmik dalğa")
    st.pyplot(fig)

mode = st.sidebar.radio("Rejim seç:", ["Real data (Replay)", "Simulyasiya (Synthetic)"])


# =====================================================
# REAL DATA MODE
# =====================================================
if mode == "Real data (Replay)":

    slices = np.load("real_slices.npy").astype("float32")
    slices = slices[:, :300]

    if st.button("Növbəti siqnalı göstər"):
        st.session_state.i += 1

    frame = slices[st.session_state.i % len(slices)]
    x = frame.reshape(1,300,1)
    score = float(model.predict(x,verbose=0)[0][0])

    st.subheader(f"Zəlzələ riski: {risk_label(score)}")
    plot_signal(frame)

    st.info(
        "Bu modda sistem real seysmik məlumatların hər bir hissəsini ardıcıllıqla təhlil edir.\n"
        "Hər dəfə 'Növbəti siqnalı göstər' düyməsinə basdıqda,\n"
        "AI modeli yeni dalğanı analiz edir və risk səviyyəsini proqnozlaşdırır."
    )


# =====================================================
# SYNTHETIC MODE
# =====================================================
else:
    mag = st.sidebar.slider("Süni dalğa gücü (Magnitude)", 3.0, 8.0, 5.0)
    noise = st.sidebar.slider("Səs-küy səviyyəsi", 0.1, 2.0, 0.5)

    if st.button("Süni siqnal yarat"):
        st.session_state.i += 1

    sig = generate_signal(mag=mag, noise_level=noise, length=300).astype("float32")
    x = sig.reshape(1,300,1)
    score = float(model.predict(x, verbose=0)[0][0])

    st.subheader(f"Zəlzələ riski: {risk_label(score)}")
    plot_signal(sig)

    st.info(
        "Bu mod AI-nın davranışını yoxlamaq üçündür.\n"
        "Magnitude və Noise səviyyəsini dəyişərək,\n"
        "AI modelinin risk proqnozunun necə dəyişdiyini müşahidə edə bilərsiniz."
    )
