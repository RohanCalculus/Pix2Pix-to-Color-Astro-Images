import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os

# Paths
test_input_folder = 'test_input'
saved_model_path = 'saved_generator_model_step_40000.h5'

IMG_WIDTH = 256
IMG_HEIGHT = 256

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(test_input_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_and_resize_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # Resize image to fixed dimensions
    image = (image * 2) - 1  # Normalize to [-1, 1]
    return image

def generate_images(model, test_input):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')  # Display grayscale image
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)  # Display colorized image
        plt.axis('off')
    plt.show()

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load the trained generator model
generator_40k = load_model(saved_model_path)

st.title("Image Colorization using GAN")

uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    
    # Read and preprocess the uploaded image
    image = load_and_resize_image(file_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    
    # Generate colorized image
    prediction = generator_40k(image, training=True)
    
    # Display input and output images
    st.write("## Input Image")
    st.image(uploaded_file, use_column_width=True, channels="GRAY")
    
    st.write("## Colorized Image")
    prediction = prediction[0].numpy() * 0.5 + 0.5  # Denormalize to [0, 1]
    st.image(prediction, use_column_width=True)

    # Cleanup: Remove the uploaded file from the test_input_folder
    os.remove(file_path)
