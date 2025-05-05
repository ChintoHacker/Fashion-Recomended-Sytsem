import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = tf.keras.models.load_model('model.h5', compile=False)

# Load Fashion MNIST
(_, _), (X_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
X_val = X_val.astype('float32') / 255.0
X_val = np.expand_dims(X_val, -1)

# Feature extraction
@st.cache_data
def extract_features(images):
    features = model.predict(images)
    return features

@st.cache_data
def calculate_similarity(features1, features2):
    return cosine_similarity(features1, features2)

def recommend(image_index, features, num_recommendations=5):
    similarities = calculate_similarity(features[image_index].reshape(1, -1), features)
    top_indices = np.argsort(-similarities[0])[:num_recommendations]
    return top_indices

# App UI
st.title("🧥 Fashion Recommendation System")
image_index = st.slider("Select an Image Index", 0, len(X_val)-1, 10)
features = extract_features(X_val)
recommended_indices = recommend(image_index, features)

st.write("### Original Image")
st.image(X_val[image_index].reshape(28, 28), width=100, caption="Original")

st.write("### Recommended Images")
cols = st.columns(len(recommended_indices))
for i, idx in enumerate(recommended_indices):
    with cols[i]:
        st.image(X_val[idx].reshape(28, 28), width=100, caption=f"Item {i+1}")
