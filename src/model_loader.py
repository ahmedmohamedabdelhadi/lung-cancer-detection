"""
model_loader.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Centralized model loading for all phases after training (Phase 5 onward).

    Save/load strategy — numpy weights:
        TF 2.10 has serialization bugs with all Keras weight formats
        (.keras, .h5, .weights.h5) when combined with ReduceLROnPlateau
        or partially frozen architectures.

        We use a numpy-based approach instead:
            Save: np.save(model.get_weights())  — plain numpy arrays
            Load: model.set_weights(np.load())  — set arrays directly

        This bypasses all format issues completely.

    Load procedure for each model:
        1. Rebuild the architecture using the original build function
        2. Unfreeze last 30 layers to match the Stage 2 saved state
        3. Call set_weights() with the loaded numpy arrays

    Also exports preprocessing functions so evaluation and inference
    notebooks can build the correct data generator for each model.

Usage:
    from src.model_loader import load_model, load_all_models
    from src.model_loader import PREPROCESSING_FUNCTIONS

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_baseline     import build_baseline_model
from src.model_efficientnet import build_efficientnet,  unfreeze_efficientnet
from src.model_inception    import build_inception,     unfreeze_inception
from src.model_resnet50     import build_resnet50,      unfreeze_resnet50

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet_preprocess

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")

# Maps model name → (build function, unfreeze function, weights filename)
# unfreeze_fn is None for baseline — it was never fine-tuned.
# Baseline uses .keras (full model save) which works fine for simple architectures.
MODEL_REGISTRY = {
    # All four models now use the same numpy strategy — no special cases
    "baseline_cnn"   : (build_baseline_model, None,                  "baseline_cnn_weights.npy",     "numpy"),
    "efficientnetb0" : (build_efficientnet,   unfreeze_efficientnet, "efficientnetb0_weights.npy",   "numpy"),
    "inceptionv3"    : (build_inception,      unfreeze_inception,    "inceptionv3_weights.npy",      "numpy"),
    "resnet50"       : (build_resnet50,        unfreeze_resnet50,     "resnet50_weights.npy",         "numpy"),
}

# Preprocessing functions per model — passed to get_data_generators()
# as preprocessing_function. None means the generator handles rescaling.
PREPROCESSING_FUNCTIONS = {
    "baseline_cnn"   : None,
    "efficientnetb0" : efficientnet_preprocess,
    "inceptionv3"    : inception_preprocess,
    "resnet50"       : resnet_preprocess,
}


def load_model(model_name, input_shape=(224, 224, 3), num_classes=3):
    """
    Rebuilds the model in its Stage 2 state and loads saved weights into it.

    For transfer learning models, weights are stored as numpy arrays
    (.npy files) saved with model.get_weights() after training.
    set_weights() restores them directly — no format conversion needed.

    For the baseline CNN, the full model is loaded from a .keras file
    (no partial-freezing issues exist for this simpler architecture).

    Args:
        model_name  (str)  : One of "baseline_cnn", "efficientnetb0",
                             "inceptionv3", "resnet50".
        input_shape (tuple): Input image shape. Default: (224, 224, 3).
        num_classes (int)  : Number of output classes. Default: 3.

    Returns:
        tf.keras.Model: Fully loaded model ready for inference.

    Raises:
        ValueError       : If model_name is not in the registry.
        FileNotFoundError: If the weights file does not exist.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    build_fn, unfreeze_fn, weights_file, fmt = MODEL_REGISTRY[model_name]
    weights_path = os.path.join(SAVED_MODELS_DIR, weights_file)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Make sure training is complete."
        )


   # Transfer learning models — rebuild + set_weights
    # Step 1: Rebuild architecture from scratch (fully frozen)
    print(f"  Building {model_name} architecture ...")
    model = build_fn(input_shape=input_shape, num_classes=num_classes)

    # Step 2: Unfreeze last 30 layers to match the Stage 2 saved state.
    # Skipped for baseline_cnn — it has no fine-tuning stage so
    # unfreeze_fn is None and the architecture is already in its final state.
    if unfreeze_fn is not None:
        print(f"  Restoring Stage 2 architecture (unfreezing last 30 layers) ...")
        model = unfreeze_fn(model, fine_tune_layers=30)

    # Step 3: Load numpy weight arrays directly into the architecture
    print(f"  Loading weights from {weights_file} ...")
    weights = np.load(weights_path, allow_pickle=True)
    model.set_weights(weights)

    print(f"  ✅ {model_name} ready.")
    return model


def load_all_models(input_shape=(224, 224, 3), num_classes=3):
    """
    Loads all four trained models and returns them as a dictionary.

    Returns:
        dict: {model_name: tf.keras.Model}
    """
    all_models = {}
    for name in MODEL_REGISTRY:
        print(f"\nLoading {name} ...")
        all_models[name] = load_model(name, input_shape, num_classes)
    return all_models