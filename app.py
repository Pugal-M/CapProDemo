
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model("asl_model.keras.h5")

# Map indices to letters/numbers (update if your dataset uses different mapping)
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(0, 10)]
@app.route("/", methods=["GET"])
def home():
    return "MNIST Predictor API is live! Use /predict to POST images."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data"}), 400

    img_array = np.array(data["image"], dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # shape (28,28,1)
    img_array = np.expand_dims(img_array, axis=0)   # shape (1,28,28,1)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_label = classes[pred_index]

    return jsonify({"prediction": pred_label})
