from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("model.h5")

# ISL class labels
class_names = ["Hello", "Yes", "No", "Thank You", "Please"]

@app.route("/", methods=["GET"])
def home():
    return "Indian Sign Language API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((64, 64))  # Change to match your training input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        return jsonify({"class": predicted_class, "confidence": confidence})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
