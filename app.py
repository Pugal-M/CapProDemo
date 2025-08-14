from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os

app = Flask(__name__)

# ✅ Load model safely
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define classes (update to match your training dataset)
class_names = ["Cat", "Dog"]

# ✅ Detect model's input size automatically
input_shape = model.input_shape[1:3]  # (height, width)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Image Recognition API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # ✅ Open and preprocess image
        img = Image.open(file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image format"}), 400

    img = img.resize(input_shape)  # auto-match model's input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch size = 1

    # ✅ Make prediction
    try:
        predictions = model.predict(img_array)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "Unknown"
    confidence = float(np.max(predictions) * 100)

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
