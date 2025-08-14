from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Convert to numpy array and reshape to match model input
        image = np.array(data['image']).reshape(1, 28, 28, 1)
        
        pred = model.predict(image)
        result = int(np.argmax(pred))  # predicted digit

        return jsonify({"prediction": result})
    
    except Exception as e:
        # Return error message to debug 500 errors
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
