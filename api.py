# api.py
import io
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Try to import optional model libraries
TF_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

app = FastAPI(title="Leaf Disease Detection API", description="Detect plant leaf diseases from images", version="1.0.0")

MODEL = None
MODEL_TYPE = None
MODEL_PATH = None

def try_load_model(project_root: str):
    """
    Try to find and load a model automatically.
    Looks for:
      - TensorFlow SavedModel dir (folder with 'saved_model.pb')
      - .h5 Keras models
      - PyTorch .pt/.pth models
    If nothing found, returns (None, None).
    """
    global MODEL, MODEL_TYPE, MODEL_PATH

    # Common model locations/filenames
    candidates = []
    for root, dirs, files in os.walk(project_root):
        # check saved_model
        if "saved_model.pb" in files:
            candidates.append(root)
        # check .h5
        for f in files:
            if f.lower().endswith(".h5"):
                candidates.append(os.path.join(root, f))
            if f.lower().endswith((".pt", ".pth")):
                candidates.append(os.path.join(root, f))

    # prefer saved_model
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "saved_model.pb")) and TF_AVAILABLE:
            try:
                MODEL = tf.keras.models.load_model(c)
                MODEL_TYPE = "tensorflow_savedmodel"
                MODEL_PATH = c
                print(f"[INFO] Loaded TensorFlow SavedModel from {c}")
                return
            except Exception as e:
                print("[WARN] Failed loading SavedModel:", e)

    # .h5
    for c in candidates:
        if isinstance(c, str) and c.lower().endswith(".h5") and TF_AVAILABLE:
            try:
                MODEL = tf.keras.models.load_model(c)
                MODEL_TYPE = "keras_h5"
                MODEL_PATH = c
                print(f"[INFO] Loaded Keras .h5 from {c}")
                return
            except Exception as e:
                print("[WARN] Failed loading .h5:", e)

    # torch
    for c in candidates:
        if isinstance(c, str) and c.lower().endswith((".pt", ".pth")) and TORCH_AVAILABLE:
            try:
                MODEL = torch.load(c, map_location="cpu")
                MODEL_TYPE = "torch"
                MODEL_PATH = c
                print(f"[INFO] Loaded PyTorch model from {c}")
                return
            except Exception as e:
                print("[WARN] Failed loading torch model:", e)

    # nothing found
    MODEL = None
    MODEL_TYPE = None
    MODEL_PATH = None
    print("[INFO] No usable model found automatically.")

# Attempt to load model on startup (project root = this file's parent)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
try_load_model(PROJECT_ROOT)

class PredictionResponse(BaseModel):
    disease_type: str
    confidence: Optional[float] = None
    analysis_timestamp: float

def preprocess_image_from_bytes(image_bytes: bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

@app.post("/disease-detection-file", response_model=PredictionResponse)
async def disease_detection_file(file: UploadFile = File(...)):
    start = time.time()
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    # quick size guard
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max ~10MB)")

    # If we have a model, try to predict
    if MODEL is not None:
        try:
            x = preprocess_image_from_bytes(contents, target_size=(224,224))
            if MODEL_TYPE in ("tensorflow_savedmodel", "keras_h5"):
                preds = MODEL.predict(x)
                # If model outputs probabilities vector
                if preds.ndim == 2 and preds.shape[1] > 1:
                    class_idx = int(np.argmax(preds[0]))
                    confidence = float(np.max(preds[0]) * 100)
                    disease_type = str(class_idx)
                else:
                    # Single-output regression / custom => return raw
                    disease_type = str(preds.tolist())
                    confidence = None
            elif MODEL_TYPE == "torch":
                # Convert to torch tensor and run inference
                import torch
                tensor = torch.from_numpy(x).permute(0,3,1,2)  # from NHWC to NCHW if needed
                MODEL.eval()
                with torch.no_grad():
                    out = MODEL(tensor)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out_np = out.cpu().numpy()
                if out_np.ndim == 2 and out_np.shape[1] > 1:
                    class_idx = int(np.argmax(out_np[0]))
                    confidence = float(np.max(out_np[0]) * 100)
                    disease_type = str(class_idx)
                else:
                    disease_type = str(out_np.tolist())
                    confidence = None
            else:
                disease_type = "unknown_model_type"
                confidence = None

            return PredictionResponse(
                disease_type=disease_type,
                confidence=round(confidence, 2) if confidence is not None else None,
                analysis_timestamp=time.time()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")
    else:
        # No model available — return dummy response with helpful message
        # You should replace this with real model loading & mapping to labels.
        return JSONResponse(status_code=200, content={
            "disease_type": "healthy (demo)",
            "confidence": 95.0,
            "analysis_timestamp": time.time(),
            "note": "No real model was found automatically. Place a TensorFlow SavedModel (folder with saved_model.pb) or a .h5 or .pt/.pth in the project root or adjust try_load_model() to point to your model."
        })
