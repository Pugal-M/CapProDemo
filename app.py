import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model("asl_model.keras.h5")

# Map indices to labels (A-Z + 0-9)
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(0, 10)]

@app.route("/", methods=["GET"])
def home():
    return "ASL Predictor API is live! Use /predict to POST an image."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    try:
        # Convert to grayscale & resize to 28x28
        img = Image.open(file).convert("L").resize((28, 28))
        img_array = np.array(img, dtype=np.float32) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1,28,28,1)

        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        pred_label = classes[pred_index]

        return jsonify({"prediction": pred_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
