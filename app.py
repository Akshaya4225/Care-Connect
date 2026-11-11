import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

from hospitals import find_hospitals

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "skin_disease_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

model = None
class_names = []

def load_artifacts():
    global model, class_names
    if MODEL_PATH.exists() and CLASS_PATH.exists():
        model = load_model(MODEL_PATH)
        class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    else:
        model = None
        class_names = []

load_artifacts()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None, class_count=len(class_names))

@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": model is not None,
        "classes": len(class_names),
        "message": "Model is ready." if model is not None else
                   "Train the model and place the generated files in the models folder."
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "The trained model is not installed yet. Follow the README training steps first."}), 503
    if "image" not in request.files:
        return jsonify({"error": "Choose an image first."}), 400

    uploaded = request.files["image"]
    if not uploaded.filename or not allowed_file(uploaded.filename):
        return jsonify({"error": "Use a PNG, JPG, JPEG or WEBP image."}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        # The model includes a Rescaling(1/255) layer, so pass pixel values in the 0-255 range.
        array = np.asarray(image, dtype=np.float32)
        probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
        order = np.argsort(probabilities)[::-1][:3]
        predictions = [
            {"label": class_names[int(i)], "confidence": round(float(probabilities[i]) * 100, 2)}
            for i in order
        ]
        return jsonify({
            "prediction": predictions[0],
            "top_predictions": predictions,
            "notice": "AI output is an educational estimate, not a medical diagnosis."
        })
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the model and class-name files."}), 500

@app.route("/api/hospitals")
def hospitals():
    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lon", ""))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError
    except ValueError:
        return jsonify({"error": "Valid latitude and longitude are required."}), 400
    try:
        return jsonify({"hospitals": find_hospitals(lat, lon)})
    except Exception:
        app.logger.exception("Hospital lookup failed")
        return jsonify({"error": "Nearby hospital search is temporarily unavailable."}), 502

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 8 MB."}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
