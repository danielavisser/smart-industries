# calcone.py
import os
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import io
import base64

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import layers, models

# bepaal de map waar calcone.py staat
BASE_DIR = os.path.dirname(__file__)
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

MODEL_PATH = "autoencoder_model.h5"
IMG_SIZE = (28, 28)  # width, height

# -------------------------
# Helpers: load and preprocess
# -------------------------
def load_and_preprocess_image(path: str) -> np.ndarray:
    """
    Load image from path, convert to grayscale, resize to IMG_SIZE,
    normalize to [0,1] and return shape (28,28,1) float32.
    """
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape((*IMG_SIZE[::-1], 1))  # (28,28,1) -> note PIL size is (w,h)
    return arr

def list_image_files() -> List[str]:
    files = [f for f in sorted(os.listdir(IMAGE_FOLDER)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return files

def load_dataset_all() -> np.ndarray:
    """
    Loads all images from IMAGE_FOLDER, preprocesses and returns array
    shape (N, 28, 28, 1)
    """
    files = list_image_files()
    imgs = []
    for f in files:
        p = os.path.join(IMAGE_FOLDER, f)
        try:
            imgs.append(load_and_preprocess_image(p))
        except Exception as e:
            print(f"Warning: kon {p} niet laden: {e}")
    if len(imgs) == 0:
        raise FileNotFoundError("Geen afbeeldingen gevonden in images/")
    data = np.stack(imgs).astype("float32")
    return data

# -------------------------
# Model: Dense autoencoder (simple / fast)
# -------------------------
def build_dense_autoencoder(latent_dim: int = 164) -> tf.keras.Model:
    input_shape = (28, 28, 1)
    inp = layers.Input(shape=input_shape)
    x = layers.Flatten()(inp)
    encoded = layers.Dense(latent_dim, activation="relu")(x)
    decoded = layers.Dense(28 * 28, activation="sigmoid")(encoded)
    out = layers.Reshape((28, 28, 1))(decoded)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Training / Save / Load
# -------------------------
def train_and_save_model(latent_dim: int = 164,
                         epochs: int = 10,
                         batch_size: int = 16,
                         validation_split: float = 0.2) -> Dict:
    """
    Loads dataset, builds model, trains, saves model to MODEL_PATH.
    Returns training history dict with 'loss' and 'val_loss'.
    """
    data = load_dataset_all()
    model = build_dense_autoencoder(latent_dim=latent_dim)
    # train (we use same data as X and y)
    history = model.fit(data, data,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=validation_split,
                        verbose=1)
    # save
    model.save(MODEL_PATH)
    return {"loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", [])}

def load_model_if_exists() -> tf.keras.Model:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model niet gevonden. Train eerst via /train endpoint.")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# -------------------------
# Prediction: for given filenames return MSE and base64 images
# -------------------------
def predict_for_files(filenames: List[str]) -> List[Dict]:
    """
    For each filename (must be present in IMAGE_FOLDER), compute reconstruction,
    MSE and return dict with keys:
      - filename
      - mse
      - original_b64 (PNG)
      - reconstructed_b64 (PNG)
    """
    model = load_model_if_exists()
    results = []

    for name in filenames:
        path = os.path.join(IMAGE_FOLDER, name)
        if not os.path.exists(path):
            results.append({"filename": name, "error": "file not found"})
            continue

        orig = load_and_preprocess_image(path)  # (28,28,1) float32
        x = np.expand_dims(orig, axis=0)  # (1,28,28,1)
        recon = model.predict(x)[0]  # (28,28,1)
        # mse
        mse = float(np.mean(np.square(orig - recon)))

        # convert to uint8 images for PNG encoding (0-255)
        orig_uint8 = (orig.squeeze() * 255.0).clip(0,255).astype("uint8")
        recon_uint8 = (recon.squeeze() * 255.0).clip(0,255).astype("uint8")

        # encode as PNG bytes and then base64
        orig_b64 = _ndarray_to_base64_png(orig_uint8)
        recon_b64 = _ndarray_to_base64_png(recon_uint8)

        results.append({
            "filename": name,
            "mse": mse,
            "original_b64": orig_b64,
            "reconstructed_b64": recon_b64
        })
    return results

def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    """
    Convert a 2D uint8 numpy array to base64-encoded PNG string.
    """
    # use PIL to save into buffer
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode("utf-8")
