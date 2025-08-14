import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image = np.array(data["image"]).reshape(1, 28, 28)
    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction))
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
