from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.h5")  # Make sure this file is in the project root

@app.route("/")  # This is the home route
def home():
    return "MNIST Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expecting JSON input
    # Example: {"image": [0.0, 0.0, ..., 0.0]}  # Flattened 28x28 image
    prediction = model.predict([data["image"]])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
