from flask import Flask, request, jsonify
import mlflow.keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
try:
    print("Setting MLflow tracking URI...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    print("Loading model from MLflow...")
    model = mlflow.keras.load_model("models:/Best_Overall_Model/2")
    print("Model loaded successfully!")

except Exception as e:
    print("Failed to load model from MLflow:", e)
    model = None  # Prevent crash

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((64, 64))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return " Waste Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()

    try:
        input_data = preprocess_image(img_bytes)
        prediction_prob = model.predict(input_data)[0][0]
        prediction_label = int(prediction_prob > 0.5)

        return jsonify({
            "prediction_probability": float(prediction_prob),
            "prediction_label": prediction_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask app on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
