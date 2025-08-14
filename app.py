import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model("asl_model.keras.h5")

# Update this mapping to match your dataset labels
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(0, 10)]

@app.route("/", methods=["GET"])
def home():
    return "ASL Predictor API is live! Use POST /predict with an image."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting base64 encoded image
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image data"}), 400

        # Decode the image from base64
        image_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))  # Resize to match your model input

        # Preprocess for model
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        pred_label = classes[pred_index]

        return jsonify({"prediction": pred_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
