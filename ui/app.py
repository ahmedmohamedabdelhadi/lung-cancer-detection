"""
app.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Gradio web interface for the Lung Cancer Detection project.

    The user uploads a lung histopathology image and receives:
        1. Predicted class with confidence percentage
        2. Confidence bar chart for all three classes
        3. Grad-CAM overlay showing which tissue regions the model focused on

    This file is self-contained — it can be run locally or deployed to
    Hugging Face Spaces in Phase 9 without any modifications.

Usage:
    From the project root:
        conda activate lung_cancer
        python ui/app.py

    Then open http://localhost:7860 in your browser.

Author : Ahmed
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — required when matplotlib
                        # is used inside a Gradio app (no display available)
import matplotlib.pyplot as plt
import cv2
import gradio as gr

from PIL import Image
import tensorflow as tf

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_loader import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE     = (224, 224)
CLASS_LABELS = ["lung_aca",            "lung_n",  "lung_scc"]
CLASS_NAMES  = ["Lung Adenocarcinoma", "Normal",  "Squamous Cell Carcinoma"]

# Color for each class — used in the confidence bar chart
CLASS_COLORS = ["#E8593C", "#1D9E75", "#7F77DD"]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — runs once at startup, not on every prediction
# ─────────────────────────────────────────────────────────────────────────────

print("Loading ResNet50 model...")
resnet_model = load_model("resnet50")
print("✅ Model loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM SETUP — build sub-models once at startup
# ─────────────────────────────────────────────────────────────────────────────

def build_gradcam_models(model):
    """
    Splits ResNet50 into a feature extractor and a logits classifier.

    We output pre-softmax logits instead of probabilities to avoid gradient
    saturation — when the model is 100% confident, softmax gradients collapse
    to ~0 and Grad-CAM produces empty heatmaps. Logits never saturate.

    Args:
        model (tf.keras.Model): Loaded ResNet50 model.

    Returns:
        tuple: (feature_model, classifier_model)
    """
    resnet_base     = model.get_layer("resnet50")
    last_conv_layer = resnet_base.get_layer("conv5_block3_out")

    # Feature extractor: outer model input → last conv output (7×7×2048)
    feature_model = tf.keras.Model(
        inputs  = resnet_base.input,
        outputs = last_conv_layer.output,
        name    = "feature_extractor"
    )

    # Classifier head: reuse trained layers with linear output (no softmax)
    gap_layer     = model.get_layer("gap")
    dropout_layer = model.get_layer("dropout")
    dense_layer   = model.get_layer("output")

    conv_input = tf.keras.Input(
        shape = last_conv_layer.output_shape[1:],
        name  = "conv_input"
    )

    x = gap_layer(conv_input)
    x = dropout_layer(x, training=False)

    # Linear activation — outputs logits, not probabilities
    logits_out = tf.keras.layers.Dense(
        units      = dense_layer.units,
        activation = None,
        name       = "logits"
    )(x)

    classifier_model = tf.keras.Model(
        inputs  = conv_input,
        outputs = logits_out,
        name    = "classifier_logits"
    )

    # Copy trained weights into the logits layer
    classifier_model.get_layer("logits").set_weights(
        dense_layer.get_weights()
    )

    return feature_model, classifier_model


print("Building Grad-CAM sub-models...")
feature_model, classifier_model = build_gradcam_models(resnet_model)
print("✅ Grad-CAM ready.")


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(pil_image):
    """
    Converts a PIL image to a preprocessed numpy array for ResNet50.

    Args:
        pil_image (PIL.Image): Image uploaded by the user.

    Returns:
        tuple: (raw_array, processed_array)
            raw_array       (np.array): uint8 RGB (224,224,3) for display.
            processed_array (np.array): float32 preprocessed (1,224,224,3)
                                        ready for model input.
    """
    # Resize to model input size and convert to RGB
    img_resized = pil_image.convert("RGB").resize(IMG_SIZE)
    raw_array   = np.array(img_resized)   # uint8 [0,255] — for display

    # Apply ResNet50 preprocessing: subtract ImageNet channel means
    processed   = raw_array.astype(np.float32)
    processed   = np.expand_dims(processed, axis=0)   # add batch dimension
    processed   = preprocess_input(processed)          # [0,255] → channel-normalized

    return raw_array, processed


def make_gradcam_heatmap(processed_img, pred_index):
    """
    Computes a Grad-CAM heatmap using pre-softmax logits.

    Runs on CPU to avoid TF 2.10 Windows cuDNN sub-model compilation error.

    Args:
        processed_img (np.array): Preprocessed image, shape (1, 224, 224, 3).
        pred_index    (int)     : Class index to explain.

    Returns:
        np.array: Normalized heatmap, shape (7, 7), values in [0, 1].
    """
    with tf.device('/CPU:0'):
        img_tensor = tf.cast(processed_img, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs = feature_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            logits       = classifier_model(conv_outputs, training=False)
            class_score  = logits[:, pred_index]

        grads        = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_outputs[0]
        heatmap  = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap  = tf.squeeze(heatmap)
        heatmap  = tf.maximum(heatmap, 0)
        heatmap  = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def create_overlay(raw_img, heatmap, alpha=0.45):
    """
    Blends a Grad-CAM heatmap with the original image.

    Args:
        raw_img (np.array): uint8 RGB image (224, 224, 3).
        heatmap (np.array): Normalized heatmap (7, 7), values in [0, 1].
        alpha   (float)   : Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        np.array: Blended uint8 RGB image (224, 224, 3).
    """
    # Resize 7×7 heatmap → 224×224
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)

    # Apply jet colormap (blue→green→yellow→red)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Alpha blend
    overlay = (alpha * heatmap_color.astype(np.float32) +
               (1 - alpha) * raw_img.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def create_confidence_chart(probabilities):
    """
    Creates a horizontal bar chart of class confidence scores.

    Args:
        probabilities (np.array): Softmax probabilities, shape (3,).

    Returns:
        matplotlib.figure.Figure: Bar chart figure for Gradio display.
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Sort by probability descending for better readability
    sorted_idx  = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_idx]
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_colors = [CLASS_COLORS[i] for i in sorted_idx]

    bars = ax.barh(
        sorted_names,
        sorted_probs * 100,
        color  = sorted_colors,
        height = 0.5,
        edgecolor = "white"
    )

    # Annotate bars with percentage
    for bar, prob in zip(bars, sorted_probs):
        ax.text(
            min(bar.get_width() + 1, 95),   # keep label inside plot
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.2f}%",
            va = "center", ha = "left",
            fontsize = 10, fontweight = "bold"
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Prediction Confidence", fontsize=11, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION — called by Gradio on every upload
# ─────────────────────────────────────────────────────────────────────────────

def predict(pil_image):
    """
    Main prediction pipeline called by the Gradio interface.

    Takes a user-uploaded image and returns:
        1. Prediction label with confidence
        2. Confidence bar chart
        3. Grad-CAM overlay image

    Args:
        pil_image (PIL.Image): Image uploaded by the user via Gradio.

    Returns:
        tuple: (label_str, confidence_chart, overlay_image)
    """
    if pil_image is None:
        return "Please upload an image.", None, None

    # ── Step 1: Preprocess ────────────────────────────────────────────────
    raw_array, processed = preprocess_image(pil_image)

    # ── Step 2: Predict ───────────────────────────────────────────────────
    with tf.device('/CPU:0'):
        probs      = resnet_model(
            tf.cast(processed, tf.float32), training=False
        ).numpy()[0]

    pred_idx    = int(np.argmax(probs))
    pred_name   = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx] * 100

    # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
    heatmap = make_gradcam_heatmap(processed, pred_idx)
    overlay = create_overlay(raw_array, heatmap)

    # ── Step 4: Confidence chart ──────────────────────────────────────────
    chart = create_confidence_chart(probs)

    # ── Step 5: Format label ──────────────────────────────────────────────
    # Medical disclaimer included — this is a research tool, not a diagnosis
    label = (
        f"Prediction  :  {pred_name}\n"
        f"Confidence  :  {confidence:.2f}%\n\n"
        f"⚠️  Research tool only — not for clinical diagnosis."
    )

    return label, chart, Image.fromarray(overlay)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def build_interface():
    """
    Builds and returns the Gradio interface.

    Layout:
        Left  column : Image upload + Predict button
        Right column : Prediction label, confidence chart, Grad-CAM overlay

    Returns:
        gr.Blocks: Configured Gradio interface ready to launch.
    """
    with gr.Blocks(title="Lung Cancer Detection") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🫁 Lung Cancer Detection
            ### CNN + Transfer Learning (ResNet50) — Test Accuracy: 99.82%

            Upload a lung histopathology image to get:
            - **Predicted class** with confidence score
            - **Confidence chart** for all three classes
            - **Grad-CAM overlay** showing which tissue regions the model focused on

            **Classes the model recognizes:**
            | Label | Meaning |
            |---|---|
            | Lung Adenocarcinoma | Malignant — glandular tissue cancer |
            | Normal | Healthy lung tissue |
            | Squamous Cell Carcinoma | Malignant — squamous cell cancer |

            > ⚠️ This tool is for research and educational purposes only.
            > It is not a substitute for clinical diagnosis by a qualified pathologist.
            """
        )

        gr.HTML("<hr>")  # horizontal separator

        # ── Main layout ───────────────────────────────────────────────────
        with gr.Row():

            # Left column — input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type    = "pil",
                    label   = "Upload Lung Histopathology Image",
                    height  = 300
                )
                predict_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant = "primary",
                    size    = "lg"
                )

            # Right column — outputs
            with gr.Column(scale=2):
                label_output = gr.Textbox(
                    label    = "Prediction Result",
                    lines    = 4,
                    max_lines = 4
                )
                chart_output = gr.Plot(
                    label = "Confidence Scores"
                )
                overlay_output = gr.Image(
                    type  = "pil",
                    label = "Grad-CAM Overlay  (Red/Yellow = high attention)"
                )

        # ── Examples ──────────────────────────────────────────────────────
        # These will be populated in Phase 9 when we add sample images
        # gr.Examples(examples=[...], inputs=image_input)

        # ── Button action ─────────────────────────────────────────────────
        predict_btn.click(
            fn      = predict,
            inputs  = [image_input],
            outputs = [label_output, chart_output, overlay_output]
        )

        # Also trigger prediction when image is uploaded directly
        image_input.change(
            fn      = predict,
            inputs  = [image_input],
            outputs = [label_output, chart_output, overlay_output]
        )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **Model:** ResNet50 fine-tuned on LC25000 dataset
            **Dataset:** 15,000 histopathology images — 3 classes × 5,000 images
            **Author:** Ahmed Mohamed Abdelhady  | Built with TensorFlow 2.10 + Gradio
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name = "0.0.0.0",   # accessible from any device on local network
        server_port = 7860,
        share       = False        # set True to get a public Gradio link
    )