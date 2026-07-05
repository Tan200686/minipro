from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import IMG_SIZE, load_model
from xai import generate_explanation_base64


MODEL_PATH = Path("hybrid_cnn_blood_cancer.h5")
CLASS_NAMES_PATH = Path("class_names.txt")


def _load_class_names() -> List[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file '{CLASS_NAMES_PATH}' not found. "
            "Train the model first (python train.py)."
        )
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


app = FastAPI(
    title="Blood Cancer Detection API",
    description="Hybrid CNN + Explainable AI (Grad-CAM) backend for blood cancer detection",
    version="1.0.0",
)

# Allow local tools / frontends to access this API if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_artifacts():
    """
    Load model and class names at application startup.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Train the model first (python train.py)."
        )

    app.state.model = load_model(str(MODEL_PATH))
    app.state.class_names = _load_class_names()


def _preprocess_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict blood cancer class probabilities for a given image.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    original_image = _preprocess_image_from_bytes(image_bytes)

    img_resized = original_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = app.state.model
    class_names = app.state.class_names

    probs = model.predict(img_array)[0]
    probs_list = probs.tolist()

    predictions = [
        {"class_name": name, "probability": float(p)}
        for name, p in zip(class_names, probs_list)
    ]

    # Sort by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predictions": predictions,
    }


@app.post("/predict-with-explanation")
async def predict_with_explanation(file: UploadFile = File(...)):
    """
    Predict and also return Grad-CAM explanation (as base64 PNG).
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    original_image = _preprocess_image_from_bytes(image_bytes)

    model = app.state.model
    class_names = app.state.class_names

    probs, heatmap_b64 = generate_explanation_base64(model, original_image)
    probs_list = probs.tolist()

    predictions = [
        {"class_name": name, "probability": float(p)}
        for name, p in zip(class_names, probs_list)
    ]
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predictions": predictions,
        "explanation": {
            "type": "grad-cam",
            "image_base64": heatmap_b64,
            "format": "png",
        },
    }


@app.get("/")
async def root():
    return {
        "message": "Blood Cancer Detection API (Hybrid CNN + Grad-CAM)",
        "endpoints": ["/predict", "/predict-with-explanation", "/docs"],
    }



