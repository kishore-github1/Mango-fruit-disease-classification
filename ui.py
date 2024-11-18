import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os 

print(f"TensorFlow version: {tf.version.VERSION}")

# Load your model
model_path = os.path.join('models', 'disease_model_h5_accurate.h5')
model = tf.keras.models.load_model(model_path)

# Define your class names (example - modify with your actual class names)
class_names = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']

# Streamlit interface
st.title("Mango fruit disease predictor")
st.write("Upload an image to get predictions.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for your model (resize and scale if necessary)
    img = img.resize((82, 82))  # Resize to your model's input size
    img_array = np.array(img) # Normalize if necessary (e.g., for models trained with ImageDataGenerator)
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Check the shape of img_array
    # st.write(f"Image shape after preprocessing: {img_array.shape}")

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # st.write(f"Image shape after expanding dimensions: {img_array.shape}")

    # Make a prediction
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class_index = np.argmax(predictions, axis=1)  # Get the index of the class with the highest probability

    # Get the class name corresponding to the predicted index
    predicted_class_name = class_names[predicted_class_index[0]]

    # Display the result
    st.write(f"Prediction: Class {predicted_class_index[0]} - {predicted_class_name}")

