import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('my_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    class_names = ['Healthy', 'Powdery', 'Rust']
    predicted_class_index = np.argmax(prediction)
    return class_names[predicted_class_index]

# Streamlit app
st.title("Plant Disease Classifier")
st.write("Upload one or multiple images to classify plant disease")

# Sidebar for image upload
uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Predict button
if st.sidebar.button('Predict'):
    if uploaded_files:
        cols = st.columns(6)  # Set up 5 columns for better distribution
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Display the image and prediction in a column
            with cols[idx % 6]:
                image = Image.open(uploaded_file)
                image = image.resize((256, 256))  # Ensure consistent image size
                st.image(image, use_column_width=True)
                
                # Make prediction
                prediction = predict_image(image, model)
                
                # Display the prediction in bold and larger font size
                st.markdown(f"<p style='font-size:20px; text-align:center;'><b>Prediction: {prediction}</b></p>", unsafe_allow_html=True)
    else:
        st.sidebar.write("Please upload at least one image.")

