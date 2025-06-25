import streamlit as st
import requests
import numpy as np
from scipy.io.wavfile import write as wav_write
from io import BytesIO
import base64

st.set_page_config(page_title="Sinteza automată a vorbirii")
st.title("Sinteza automată a vorbirii")

text = st.text_area("Introduceți textul pentru sinteza automată a vorbirii:", height=150)

if st.button("Generează audio"):
    if not text.strip():
        st.warning("Vă rugăm să introduceți un text.")
    else:
        with st.spinner("Se generează audio..."):
            try:
                response = requests.post(
                    "http://localhost:7750/generate-audio",
                    params={"text": text}
                )

                if response.status_code == 200:
                    audio_json = response.content
                    audio_np = np.frombuffer(audio_json, dtype=np.float32)

                    sample_rate = 24000

                    audio_int16 = np.int16(audio_np * 32767)

                    wav_io = BytesIO()
                    wav_write(wav_io, sample_rate, audio_int16)
                    wav_io.seek(0)
                    wav_bytes = wav_io.read()

                    b64_audio = base64.b64encode(wav_bytes).decode()
                    audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                        Browserul dvs. nu suportă redarea audio.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                    st.audio(wav_bytes, format="audio/wav")
                    st.download_button(
                        label="Descarcă audio",
                        data=wav_bytes,
                        file_name="sinteza.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error(f"Eroare la generarea audio: {response.status_code}")
            except Exception as e:
                st.error(f"A apărut o eroare: {e}")
