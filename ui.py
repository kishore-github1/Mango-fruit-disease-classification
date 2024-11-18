import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# App title and layout
st.set_page_config(
    page_title="Mango Disease Predictor",
    page_icon="🍋",
    layout="wide"
)

# Sidebar with app info
st.sidebar.title("About the App 🍋")
st.sidebar.write("""
This app predicts diseases in mango fruits using a deep learning model.  
Upload an image of a mango, and the app will analyze it for common diseases.
""")
# st.sidebar.write("### Developed by: Your Name")
# st.sidebar.write("[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)")

# Main title
st.markdown("<h1 style='text-align: center; color: green;'>Mango Disease Predictor</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Upload an image of a mango to get started.</p>", unsafe_allow_html=True)

# Load your model
model_path = os.path.join('models', 'disease_model_h5_accurate.h5')
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image',  width=300)

    # Process the image
    with st.spinner("Analyzing the image..."):
        img = img.resize((82, 82))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class_name = class_names[predicted_class_index[0]]

    # Display prediction
    st.success("Prediction Complete!")
    st.markdown(f"<h3 style='text-align: center; color: blue;'>Prediction: {predicted_class_name}</h3>", unsafe_allow_html=True)

    # Add a progress bar for probability
    confidence = predictions[0][predicted_class_index[0]]
    st.progress(int(confidence * 100))
    st.write(f"**Confidence Level**: {confidence:.2f}")

    # Download result
    result_text = f"The mango is predicted to have: {predicted_class_name} ({confidence:.2f} confidence)"
    st.download_button(label="Download Prediction", data=result_text, file_name="prediction.txt", mime="text/plain")
else:
    st.info("Please upload an image to start the prediction.")
