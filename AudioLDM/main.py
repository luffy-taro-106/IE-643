import streamlit as st
from PIL import Image
import torch
import tempfile
import os
import time
from audioldm import build_model  # same as your script
from inference_module import generate_audio_from_image  # we'll define this

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Image-to-Audio Generator", layout="centered")
st.title("🎨→🔊 Image-to-Audio Inference")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    # Preview the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to start inference
    if st.button("Generate Audio"):
        progress = st.progress(0)
        status_text = st.empty()

        # Create temp file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            image_path = tmp_img.name

        # Load model (cached so it doesn’t reload every time)
        @st.cache_resource
        def load_model():
            model_path = "./ckpt/audioldm-s-full.ckpt"
            return build_model(ckpt_path=model_path)

        model = load_model()

        # Simulate progress
        for i in range(0, 100, 10):
            time.sleep(0.2)
            progress.progress(i)
            status_text.text(f"Generating... {i}%")

        # Run your inference function
        output_audio_path = generate_audio_from_image(model, image_path)

        progress.progress(100)
        status_text.text("✅ Done!")

        # Play the generated audio
        audio_file = open(output_audio_path, "rb")
        st.audio(audio_file.read(), format="audio/wav")
