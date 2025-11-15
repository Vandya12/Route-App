import streamlit as st
import numpy as np
from PIL import Image
from utils.predict import load_segmentation_model, predict_mask

st.set_page_config(page_title="Highway Segmentation App", layout="wide")
st.title("🛰️ Satellite Image Segmentation (Highway Project)")

@st.cache_resource
def load_model_cached():
    return load_segmentation_model()

model = load_model_cached()

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    st.image(img_np, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Segmentation"):
        mask = predict_mask(model, img_np)
        st.image(mask, caption="Predicted Mask", use_column_width=True)
