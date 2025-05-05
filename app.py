import numpy as np
import tensorflow as tf
import gradio as gr
from sklearn.neighbors import NearestNeighbors

# Load pre-trained model
model = tf.keras.models.load_model("fashion_model_with_attention.h5")

# Create embedding model (output from dense layer)
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer(index=-4).output  # Gets the 128-dim dense layer
)

# Load Fashion MNIST data for recommendations
(_, _), (X_all, y_all) = tf.keras.datasets.fashion_mnist.load_data()
X_all = X_all.astype('float32') / 255.0
X_all = np.expand_dims(X_all, -1)

# Precompute embeddings for all images
all_embeddings = embedding_model.predict(X_all)

# Build Nearest Neighbors index
nbrs = NearestNeighbors(n_neighbors=5).fit(all_embeddings)

# Class names for Fashion MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def recommend_images(input_data, input_type):
    # Process input
    if input_type == "index":
        if input_data >= 10000:
            raise gr.Error("Index must be < 10000 (test set size)")
        input_image = X_all[input_data]
        true_label = y_all[input_data]
    else:  # uploaded image
        # Convert to 28x28 grayscale
        input_image = tf.image.rgb_to_grayscale(input_data)
        input_image = tf.image.resize(input_image, (28, 28))
        input_image = input_image.numpy().astype('float32') / 255.0
        input_image = np.expand_dims(input_image, -1)
        true_label = "Uploaded Image"

    # Get embedding and find neighbors
    emb = embedding_model.predict(np.array([input_image]))
    distances, indices = nbrs.kneighbors(emb)
    
    # Prepare images and labels
    input_image = (input_image * 255).astype('uint8').squeeze()
    recommendations = [
        (X_all[i].squeeze(), f"Class: {class_names[y_all[i]]}")
        for i in indices[0]
    ]
    
    return input_image, recommendations, f"True class: {class_names[true_label] if input_type == 'index' else true_label}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👗 Fashion Recommender System")
    
    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(
                ["index", "upload"],
                label="Input Type",
                value="index"
            )
            index_input = gr.Number(
                label="Enter Image Index (0-9999)",
                value=0,
                visible=True
            )
            upload_input = gr.Image(
                label="Upload Image",
                visible=False
            )
            submit_btn = gr.Button("Recommend")
            
        with gr.Column():
            original_image = gr.Image(
                label="Original Image",
                shape=(28, 28)
            )
            true_label = gr.Label(label="Prediction")
            
    gr.Markdown("## Recommended Items")
    gallery = gr.Gallery(
        label="Similar Items",
        columns=5,
        object_fit="contain"
    )

    def toggle_inputs(input_type):
        return {
            index_input: gr.update(visible=input_type == "index"),
            upload_input: gr.update(visible=input_type == "upload")
        }
    
    input_type.change(
        toggle_inputs,
        inputs=input_type,
        outputs=[index_input, upload_input]
    )
    
    submit_btn.click(
        recommend_images,
        inputs=[gr.combine(lambda it, idx, img: idx if it == "index" else img, [input_type, index_input, upload_input]), input_type],
        outputs=[original_image, gallery, true_label]
    )

demo.launch()