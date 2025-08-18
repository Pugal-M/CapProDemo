# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Example class names (change as per dataset)
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

@app.route("/", methods=["GET"])
def home():
    return "🚀 Image Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting image file
        file = request.files["file"]
        img = Image.open(file).resize((32, 32))  # Resize to model input
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "class": class_names[class_index],
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)})
