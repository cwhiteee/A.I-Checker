
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from detector import detect

st.set_page_config(page_title="AI Detector", page_icon="🤖")
st.title("🤖 AI Image Detector — Heuristic v2")
st.write("Upload a photo to check if it might be **AI-generated** or **camera-captured**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def verdict(p):
    if p >= 0.75:
        return "Very likely AI-generated"
    if p >= 0.6:
        return "Likely AI-generated"
    if p >= 0.4:
        return "Uncertain / Mixed signals"
    return "Likely camera-captured"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing..."):
        ai_prob, features, debug = detect(image)

    st.subheader("Result")
    st.metric("AI likelihood", f"{ai_prob*100:.1f}%")
    st.write(f"**Verdict:** {verdict(ai_prob)}")

    with st.expander("See feature breakdown"):
        for k, v in debug["feature_scores"].items():
            st.write(f"- **{k}**: {v:.2f}")
        st.caption("Values closer to 1.0 indicate the feature looked more like AI.")

    st.info(
        "This detector uses multiple handcrafted signals: missing EXIF, facial symmetry, "
        "micro-texture vs background smoothness, frequency spectrum peakiness, and color distribution. "
        "It's heuristic (no cloud model) so use results as guidance, not proof."
    )

st.caption("No images are uploaded to any server; all analysis runs locally in this app's process.")
