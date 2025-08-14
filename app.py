from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Class names (update these as per your dataset)
class_names = ["Cat", "Dog"]

@app.route("/", methods=["GET"])
def home():
    return "Image Recognition API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))  # Change to your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
