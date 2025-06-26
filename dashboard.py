import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/best_cnn_model_overall.keras")
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((64, 64)).convert('RGB')
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("Trash vs Recyclable Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)[0][0]

    # prediction is sigmoid output; threshold at 0.5
    if prediction >= 0.5:
        st.success(f"Prediction: Recyclable (score: {prediction:.2f})")
    else:
        st.error(f"Prediction: Trash (score: {prediction:.2f})")