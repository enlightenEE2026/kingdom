# -*- coding: utf-8 -*-

import io
import os
import sys
import subprocess
import importlib

# 1. DEPENDENCY MANAGEMENT
def ensure_libraries():
    required_packages = {
        "easyocr": "easyocr",
        "PyPDF2": "PyPDF2",
        "thefuzz": "thefuzz[speed]",
        "PIL": "pillow",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "streamlit": "streamlit",
    }
    for module_name, pip_name in required_packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            except subprocess.CalledProcessError:
                sys.exit(1)

ensure_libraries()

# Import required tools after ensuring they are installed
import streamlit as st
import easyocr
import PyPDF2
from PIL import Image, ImageDraw, ImageFont
from thefuzz import process, fuzz
import numpy as np

# 2. CACHE OCR READER (Prevents reloading on every user interaction)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ch_sim', 'en'])

reader = load_ocr_reader()

# 3. STREAMLIT APP LAYOUT
st.set_page_config(layout="wide")
st.title("🌐 Image Translation Overlay Tool")

st.markdown("### 1. Upload Files & Adjust Sensitivity")

# Create layout columns for file uploaders
col1, col2 = st.columns(2)
with col1:
    uploaded_image_file = st.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"])
with col2:
    uploaded_pdf_file = st.file_uploader("Upload PDF Translation Dictionary", type=["pdf"])

# Configuration settings
user_threshold = st.slider('Match Sensitivity %:', min_value=30, max_value=100, value=60, step=5)

st.markdown("### 2. Execute Overlay")
execute_btn = st.button("Translate and Overlay Text", type="primary")

# 4. TRANSLATION PROCESSING ENGINE
if execute_btn:
    if not uploaded_image_file:
        st.error("Please upload an image file.")
    elif not uploaded_pdf_file:
        st.error("Please upload your PDF translation file.")
    else:
        with st.spinner("Parsing PDF dictionary and scanning image text... Please wait..."):
            try:
                # --- PARSE PDF TRANSLATION DICTIONARY ---
                pdf_data = uploaded_pdf_file.read()
                pdf_file = io.BytesIO(pdf_data)
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                translation_map = {}
                excel_choices = []

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if not page_text:
                        continue

                    lines = page_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = []
                        if ',' in line:
                            parts = line.split(',', 1)
                        elif ':' in line:
                            parts = line.split(':', 1)
                        else:
                            for idx, char in enumerate(line):
                                if idx > 0 and ord(char) < 128 and ord(line[idx-1]) >= 128:
                                    parts = [line[:idx].strip(), line[idx:].strip()]
                                    break

                        if len(parts) == 2:
                            source_text = parts[0].strip()
                            translation_text = parts[1].strip()

                            if source_text and translation_text:
                                translation_map[source_text.lower()] = translation_text
                                excel_choices.append(source_text)

                if not translation_map:
                    st.error("Could not extract any valid Source/Translation pairs from the PDF layout.")
                    st.info("Ensure your PDF format maps terms cleanly (e.g., '张飞, Zhang Fei' or '张飞: Zhang Fei').")
                else:
                    # --- PROCESS AND OVERLAY TARGET IMAGE ---
                    image_data = uploaded_image_file.read()
                    original_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
                    base_image = original_image.copy()

                    ocr_results = reader.readtext(np.array(base_image), detail=1)

                    if not ocr_results:
                        st.warning("No text blocks detected in the image.")
                    else:
                        draw_layer = ImageDraw.Draw(base_image)
                        font = ImageFont.load_default()
                        matches_count = 0

                        for box, text_raw, confidence in ocr_results:
                            text_clean = text_raw.strip()
                            if not text_clean:
                                continue

                            best_match, score = process.extractOne(text_clean, excel_choices, scorer=fuzz.token_sort_ratio)

                            if score >= user_threshold:
                                translation_text = str(translation_map[best_match.lower()])
                                matches_count += 1

                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                                draw_layer.rectangle([x_min, y_min, x_max, y_max], fill=(15, 15, 15, 245))
                                draw_layer.text((x_min + 4, y_min + 2), translation_text, fill=(255, 255, 255, 255), font=font)

                        st.success(f"✨ Successfully replaced {matches_count} text blocks!")

                        # --- DISPLAY IMAGES SIDE BY SIDE IN STREAMLIT ---
                        out_col1, out_col2 = st.columns(2)
                        with out_col1:
                            st.subheader("Original Image")
                            st.image(original_image, use_container_width=True)
                        with out_col2:
                            st.subheader("Translated Overlay")
                            st.image(base_image, use_container_width=True)

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
