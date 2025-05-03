import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('fashion_model.h5')

# Load and preprocess Fashion MNIST data
@st.cache_data
def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    return X_test, y_test

X_test, y_test = load_data()

# Feature extractor
feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features
@st.cache_data
def extract_features(images):
    features = feature_extractor.predict(images)
    return features

features = extract_features(X_test)

# Similarity calculator
def calculate_similarity(features1, features2):
    return cosine_similarity(features1, features2)

# Recommend items
def recommend_fashion_item(image_index, num_recommendations=5):
    similarities = calculate_similarity(features[image_index].reshape(1, -1), features)
    top_indices = np.argsort(-similarities[0])[1:num_recommendations+1]
    return top_indices

# Streamlit UI
st.title("Fashion MNIST Similarity Recommender")
st.markdown("Select an image index to view recommendations based on visual similarity.")

image_index = st.slider("Select image index (0-9999)", 0, len(X_test)-1, 10)
num_recommendations = st.slider("Number of recommendations", 1, 10, 5)

recommended_indices = recommend_fashion_item(image_index, num_recommendations)

st.subheader("Query Image")
st.image(X_test[image_index].reshape(28, 28), width=150, caption="Query")

st.subheader("Recommended Items")
cols = st.columns(num_recommendations)
for i, idx in enumerate(recommended_indices):
    with cols[i]:
        st.image(X_test[idx].reshape(28, 28), width=150, caption=f"Rec {i+1}")
