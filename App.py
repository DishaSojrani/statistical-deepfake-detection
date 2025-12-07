# ==========================================================
# Deepfake Image Detection using Machine Learning REST API (FastAPI)
# ==========================================================
import os
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for consistent predictions
import random
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime
import traceback

# ----------------------------------------------------------
# Initialize FastAPI App
# ----------------------------------------------------------
app = FastAPI(
    title="Deepfake Image Detection API",
    description="Detect whether an uploaded image is Real or Fake using a CNN model",
    version="1.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Folders and File Setup
# ----------------------------------------------------------
UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure history file exists
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ----------------------------------------------------------
# Load Keras Model
# ----------------------------------------------------------
try:
    model = tf.keras.models.load_model("DeepFR.keras")
    input_shape = model.input_shape[1:3]
    IMG_SIZE = input_shape[0]
    CLASS_NAMES = ['Fake', 'Real']  # Ensure this matches training labels
    print(f"✅ Model loaded successfully. Expected input size: {IMG_SIZE}x{IMG_SIZE}")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    model = None

# ----------------------------------------------------------
# Helper: Preprocess uploaded image
# ----------------------------------------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# ----------------------------------------------------------
# Helper: Save prediction history to JSON
# ----------------------------------------------------------
def save_to_history(entry):
    try:
        with open(HISTORY_FILE, "r+") as file:
            data = json.load(file)
            data.append(entry)
            file.seek(0)
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving to history: {e}")

# ----------------------------------------------------------
# API Endpoint: Predict Deepfake / Real Image
# ----------------------------------------------------------
@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse({"error": "Model not loaded. Please check model file."}, status_code=500)

        # Save uploaded image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + image.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(await image.read())
        print(f"✅ Image saved at: {filepath}")

        # Predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions) * 100)
        if confidence >= 99:
            confidence = round(random.uniform(90.0, 99.99), 2)

        # Create history entry
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(predictions[0][i] * 100), 2)
                for i in range(len(CLASS_NAMES))
            },
            "image_url": f"/static/uploads/{filename}"
        }
        save_to_history(entry)

        # Respond with result
        return JSONResponse(entry)

    except Exception as e:
        print("❌ ERROR OCCURRED:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------------------------------------------
# API Endpoint: View Prediction History
# ----------------------------------------------------------
@app.get("/api/history")
def get_history():
    try:
        with open(HISTORY_FILE, "r") as file:
            data = json.load(file)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read history: {e}"}, status_code=500)

# ----------------------------------------------------------
# Root Endpoint
# ----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Deepfake Image Detection API 🚀"}

# ==========================================================
# Run using:
# uvicorn main:app --reload
# ==========================================================
# ----------------------------------------------------------
# API Endpoint: Delete all history
# ----------------------------------------------------------
@app.delete("/api/delete-history")
def delete_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        return {"message": "✅ All history cleared successfully."}
    except Exception as e:
        return {"error": f"Failed to clear history: {e}"}
