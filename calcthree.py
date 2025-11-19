# calcthree.py
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Autoencoder Demo", layout="wide")
st.title("Autoencoder Demo — reconstructies & MSE")

# --- 1) Fetch image list from backend
@st.cache_data(ttl=300)
def fetch_image_list():
    try:
        r = requests.get(f"{API_BASE}/list_images/")
        r.raise_for_status()
        return r.json().get("images", [])
    except Exception as e:
        st.error(f"Kon images lijst niet ophalen: {e}")
        return []

images = fetch_image_list()

if not images:
    st.warning("Geen afbeeldingen gevonden in backend. Zorg dat 'images/' bestaat en de API draait.")
    st.stop()

# multi-select (checkbox-style dropdown)
selected = st.multiselect("Selecteer afbeeldingen voor reconstructie", images, default=images[:5])

# Training controls
st.sidebar.header("Train model")
latent_dim = st.sidebar.number_input("latent_dim", min_value=8, max_value=1024, value=164, step=8)
epochs = st.sidebar.number_input("epochs", min_value=1, max_value=200, value=5)
batch_size = st.sidebar.number_input("batch_size", min_value=1, max_value=256, value=16)

if st.sidebar.button("Train model (backend)"):
    with st.spinner("Training gestart... dit kan even duren"):
        try:
            payload = {
                "latent_dim": int(latent_dim),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "validation_split": 0.2
            }
            r = requests.post(f"{API_BASE}/train/", json=payload, timeout=3600)
            r.raise_for_status()
            res = r.json()
            st.success("Training voltooid en model opgeslagen op backend.")
            hist = res.get("history", {})
            st.write("Laatste loss:", hist.get("loss", [])[-1] if hist.get("loss") else None)
        except Exception as e:
            st.error(f"Training failed: {e}")

# Predict button
if st.button("Run reconstruction on selected images"):
    if not selected:
        st.warning("Selecteer eerst ten minste 1 afbeelding.")
    else:
        with st.spinner("Vraag reconstructies op van backend..."):
            try:
                r = requests.post(f"{API_BASE}/predict/", json={"filenames": selected}, timeout=120)
                r.raise_for_status()
                data = r.json().get("results", [])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                data = []

        if data:
            # Show results in grid: for each file show original, recon and mse
            cols_per_image = 3
            n = len(data)
            rows = (n + 0)  # one row that scrolls horizontally via Streamlit layout columns
            # We'll create columns for each image triple
            cols = st.columns(n)
            for idx, item in enumerate(data):
                col = cols[idx]
                with col:
                    fname = item.get("filename")
                    if item.get("error"):
                        st.error(f"{fname}: {item.get('error')}")
                        continue
                    mse = item.get("mse")
                    orig_b64 = item.get("original_b64")
                    recon_b64 = item.get("reconstructed_b64")

                    # decode b64 to bytes and display
                    orig_bytes = base64.b64decode(orig_b64)
                    recon_bytes = base64.b64decode(recon_b64)

                    st.markdown(f"**{fname}**")
                    st.image(orig_bytes, caption=f"Origineel\nMSE={mse:.6f}", use_column_width=True)
                    st.image(recon_bytes, caption="Reconstructie", use_column_width=True)

            # Also show a table of filename & mse
            st.subheader("MSE per afbeelding")
            table_data = [{"filename": it["filename"], "mse": it.get("mse")} for it in data]
            st.table(table_data)
