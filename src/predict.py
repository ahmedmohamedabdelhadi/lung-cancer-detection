"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Standalone prediction script for the Lung Cancer Detection project.

    Loads the best model (ResNet50) and predicts the class of a single
    lung histopathology image from the command line.

Usage:
    From the project root:
        conda activate lung_cancer
        python src/predict.py --image path/to/image.jpg

    Optional arguments:
        --image      Path to input image (required)
        --model      Model to use: resnet50, efficientnetb0, inceptionv3,
                     baseline_cnn (default: resnet50)

Example:
    python src/predict.py --image data/lung_image_sets/lung_aca/lungaca1.jpeg
    python src/predict.py --image my_image.jpg --model efficientnetb0

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.model_loader import load_model
from PIL import Image

from tensorflow.keras.applications.resnet50     import preprocess_input as res_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Lung Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

# Maps model name → preprocessing function
# baseline_cnn uses simple rescaling (handled inside preprocess_image)
PREPROCESS_FN = {
    "resnet50"       : res_preprocess,
    "efficientnetb0" : eff_preprocess,
    "inceptionv3"    : inc_preprocess,
    "baseline_cnn"   : None,
}


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict the class of a lung histopathology image."
    )
    parser.add_argument(
        "--image",
        type     = str,
        required = True,
        help     = "Path to the input image file."
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "resnet50",
        choices = ["resnet50", "efficientnetb0", "inceptionv3", "baseline_cnn"],
        help    = "Model to use for prediction (default: resnet50)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path, model_name):
    """
    Loads and preprocesses an image for the specified model.

    Args:
        image_path (str): Path to the input image.
        model_name (str): Model name — determines preprocessing applied.

    Returns:
        np.array: Preprocessed image, shape (1, 224, 224, 3).

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load, resize, convert to RGB
    img      = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_arr  = np.array(img, dtype=np.float32)
    img_arr  = np.expand_dims(img_arr, axis=0)   # add batch dimension → (1, 224, 224, 3)

    preprocess_fn = PREPROCESS_FN[model_name]

    if preprocess_fn is not None:
        # Transfer learning models: apply model-specific normalization
        img_arr = preprocess_fn(img_arr)
    else:
        # Baseline CNN: simple rescaling to [0, 1]
        img_arr = img_arr / 255.0

    return img_arr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main entry point — loads model, preprocesses image, and prints prediction.
    """
    args = parse_args()

    print("=" * 55)
    print("  Lung Cancer Detection — Single Image Prediction")
    print("  Author: Ahmed Mohamed Abdelhady")
    print("=" * 55)
    print(f"  Image : {args.image}")
    print(f"  Model : {args.model}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(args.model)

    # ── Preprocess image ──────────────────────────────────────────────────
    print("Preprocessing image...")
    img_array = preprocess_image(args.image, args.model)

    # ── Predict ───────────────────────────────────────────────────────────
    print("Running prediction...\n")
    probs      = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    # ── Print results ─────────────────────────────────────────────────────
    print("─" * 55)
    print("  PREDICTION RESULT")
    print("─" * 55)
    print(f"  Predicted class : {pred_class}")
    print(f"  Confidence      : {confidence:.2f}%")
    print()
    print("  All class probabilities:")
    for cls_name, prob in zip(CLASS_NAMES, probs):
        bar    = "█" * int(prob * 30)
        print(f"  {cls_name:<30} {prob*100:>6.2f}%  {bar}")
    print("─" * 55)
    print("\n⚠️  Research tool only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()