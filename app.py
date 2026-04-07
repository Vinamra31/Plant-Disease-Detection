import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to detect if it's **healthy or diseased**.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing..."):
            try:
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                res = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("image.png", img_bytes, "image/png")}
                )
                result = res.json()

                if res.status_code == 200:
                    st.success(f"🌱 Prediction: **{result['prediction']}**")
                    st.info(f"Confidence: **{result['confidence']}%**")
                else:
                    st.error(f"Error: {result['detail']}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")