"""
train.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Standalone training script for the Lung Cancer Detection project.

    Trains all four models in sequence:
        1. Baseline CNN (custom architecture, no pretrained weights)
        2. EfficientNetB0 (transfer learning, two-stage fine-tuning)
        3. InceptionV3    (transfer learning, two-stage fine-tuning)
        4. ResNet50       (transfer learning, two-stage fine-tuning)

    All trained weights are saved to saved_models/ as .npy files.
    Training curves are saved to results/.

Usage:
    From the project root:
        conda activate lung_cancer
        python src/train.py

    Optional arguments:
        --data_dir   Path to dataset root  (default: data/lung_image_sets)
        --epochs     Max epochs per model  (default: 30)
        --batch_size Batch size            (default: 32)

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_preprocessing  import get_data_generators, IMG_SIZE, BATCH_SIZE
from src.model_baseline      import build_baseline_model
from src.model_efficientnet  import build_efficientnet,  unfreeze_efficientnet
from src.model_inception     import build_inception,     unfreeze_inception
from src.model_resnet50      import build_resnet50,      unfreeze_resnet50

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
from tensorflow.keras.applications.resnet50     import preprocess_input as res_preprocess


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train all lung cancer detection models."
    )
    parser.add_argument(
        "--data_dir",
        type    = str,
        default = "data/lung_image_sets",
        help    = "Path to dataset root folder (default: data/lung_image_sets)"
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 30,
        help    = "Maximum training epochs per model (default: 30)"
    )
    parser.add_argument(
        "--batch_size",
        type    = int,
        default = 32,
        help    = "Batch size (default: 32)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks():
    """
    Returns a fresh set of training callbacks.

    Must be called separately for each model to reset internal state.
    ModelCheckpoint is intentionally excluded — we save weights manually
    using numpy after training to avoid TF 2.10 serialization bugs.

    Returns:
        list: [EarlyStopping, ReduceLROnPlateau]
    """
    return [
        EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SAVE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def save_weights_numpy(model, model_name, save_dir):
    """
    Saves model weights as a numpy .npy file.

    TF 2.10 has serialization bugs with all Keras weight formats when
    ReduceLROnPlateau modifies the learning rate. Numpy save bypasses
    all format issues completely.

    EarlyStopping with restore_best_weights=True has already restored
    the best weights into model memory before this is called.

    Args:
        model      (tf.keras.Model): Trained model with best weights loaded.
        model_name (str)           : Used to name the .npy file.
        save_dir   (str)           : Directory to save weights into.
    """
    os.makedirs(save_dir, exist_ok=True)
    path    = os.path.join(save_dir, f"{model_name}_weights.npy")
    weights = model.get_weights()
    np.save(path, np.array(weights, dtype=object), allow_pickle=True)
    print(f"  💾 Weights saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history_s1, history_s2, model_name, results_dir):
    """
    Plots combined Stage 1 + Stage 2 training curves and saves to results/.

    Args:
        history_s1   : History object from Stage 1 model.fit().
        history_s2   : History object from Stage 2 model.fit().
                       Pass None for baseline (single-stage training).
        model_name   : Display name for plot title.
        results_dir  : Directory to save the plot image.
    """
    os.makedirs(results_dir, exist_ok=True)

    if history_s2 is not None:
        # Concatenate Stage 1 + Stage 2 metrics
        acc      = history_s1.history["accuracy"]     + history_s2.history["accuracy"]
        val_acc  = history_s1.history["val_accuracy"] + history_s2.history["val_accuracy"]
        loss     = history_s1.history["loss"]         + history_s2.history["loss"]
        val_loss = history_s1.history["val_loss"]     + history_s2.history["val_loss"]
        stage2_start = len(history_s1.history["accuracy"]) + 1
    else:
        acc      = history_s1.history["accuracy"]
        val_acc  = history_s1.history["val_accuracy"]
        loss     = history_s1.history["loss"]
        val_loss = history_s1.history["val_loss"]
        stage2_start = None

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_m, val_m, title, ylabel in zip(
        axes,
        [acc,  loss],
        [val_acc,  val_loss],
        [f"{model_name} — Accuracy", f"{model_name} — Loss"],
        ["Accuracy", "Loss"]
    ):
        ax.plot(epochs, train_m, color="#378ADD", linewidth=2, label="Training")
        ax.plot(epochs, val_m,   color="#E8593C", linewidth=2,
                linestyle="--", label="Validation")

        if stage2_start:
            ax.axvline(x=stage2_start, color="#1D9E75",
                       linestyle=":", linewidth=1.5, label="Fine-tuning starts")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(
        results_dir,
        f"{model_name.lower().replace(' ', '_')}_training_curves.png"
    )
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  💾 Training curves saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(data_dir, epochs, batch_size, save_dir, results_dir):
    """
    Trains the baseline CNN from scratch and saves weights.

    Args:
        data_dir    (str): Path to dataset root.
        epochs      (int): Maximum training epochs.
        batch_size  (int): Batch size.
        save_dir    (str): Directory to save model weights.
        results_dir (str): Directory to save training curve plots.

    Returns:
        float: Best validation accuracy achieved.
    """
    print("\n" + "=" * 60)
    print("  Training Baseline CNN")
    print("=" * 60)

    train_gen, val_gen, _ = get_data_generators(
        data_dir   = data_dir,
        batch_size = batch_size,
        rescale    = True
    )

    model = build_baseline_model(input_shape=(224, 224, 3), num_classes=3)

    history = model.fit(
        train_gen,
        epochs           = epochs,
        steps_per_epoch  = train_gen.n // batch_size,
        validation_data  = val_gen,
        validation_steps = val_gen.n  // batch_size,
        callbacks        = get_callbacks(),
        verbose          = 1
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\n  Best validation accuracy : {val_acc * 100:.2f}%")

    save_weights_numpy(model, "baseline_cnn", save_dir)
    plot_history(history, None, "Baseline CNN", results_dir)

    return val_acc


def train_transfer_model(
    model_name, build_fn, unfreeze_fn, preprocess_fn,
    data_dir, epochs, batch_size, save_dir, results_dir
):
    """
    Trains a transfer learning model using two-stage fine-tuning.

    Stage 1 — Feature extraction: frozen base, head only (5 epochs)
    Stage 2 — Fine-tuning: last 30 layers unfrozen (up to epochs - 5)

    Args:
        model_name    (str)     : Display name, e.g. "EfficientNetB0".
        build_fn      (callable): Function that builds the frozen model.
        unfreeze_fn   (callable): Function that unfreezes last 30 layers.
        preprocess_fn (callable): Model-specific preprocessing function.
        data_dir      (str)     : Path to dataset root.
        epochs        (int)     : Maximum total training epochs.
        batch_size    (int)     : Batch size.
        save_dir      (str)     : Directory to save model weights.
        results_dir   (str)     : Directory to save training curve plots.

    Returns:
        float: Best validation accuracy achieved.
    """
    print("\n" + "=" * 60)
    print(f"  Training {model_name}")
    print("=" * 60)

    train_gen, val_gen, _ = get_data_generators(
        data_dir               = data_dir,
        batch_size             = batch_size,
        rescale                = False,
        preprocessing_function = preprocess_fn
    )

    model = build_fn(input_shape=(224, 224, 3), num_classes=3)

    # ── Stage 1: Feature extraction (5 epochs) ────────────────────────────
    print(f"\n  {model_name} — Stage 1: Feature extraction (5 epochs)")
    history_s1 = model.fit(
        train_gen,
        epochs           = 5,
        steps_per_epoch  = train_gen.n // batch_size,
        validation_data  = val_gen,
        validation_steps = val_gen.n  // batch_size,
        callbacks        = get_callbacks(),
        verbose          = 1
    )

    # ── Stage 2: Fine-tuning ──────────────────────────────────────────────
    print(f"\n  {model_name} — Stage 2: Fine-tuning (up to {epochs - 5} epochs)")
    model = unfreeze_fn(model, fine_tune_layers=30)

    history_s2 = model.fit(
        train_gen,
        epochs           = epochs - 5,
        steps_per_epoch  = train_gen.n // batch_size,
        validation_data  = val_gen,
        validation_steps = val_gen.n  // batch_size,
        callbacks        = get_callbacks(),
        verbose          = 1
    )

    val_acc = max(history_s2.history["val_accuracy"])
    print(f"\n  Best validation accuracy : {val_acc * 100:.2f}%")

    save_weights_numpy(model, model_name.lower().replace(" ", ""), save_dir)
    plot_history(history_s1, history_s2, model_name, results_dir)

    return val_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():  # sourcery skip: extract-duplicate-method, merge-dict-assign
    """
    Main entry point — trains all four models sequentially and prints
    a final comparison table.
    """
    args        = parse_args()
    save_dir    = "saved_models"
    results_dir = "results"

    print("=" * 60)
    print("  Lung Cancer Detection — Training Pipeline")
    print("  Author: Ahmed Mohamed Abdelhady")
    print("=" * 60)
    print(f"  Data directory : {args.data_dir}")
    print(f"  Max epochs     : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")

    results = {}

    # ── Train all four models ─────────────────────────────────────────────
    results["Baseline CNN"] = train_baseline(
        data_dir    = args.data_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        save_dir    = save_dir,
        results_dir = results_dir
    )

    for model_name, build_fn, unfreeze_fn, preprocess_fn in [
        ("EfficientNetB0", build_efficientnet, unfreeze_efficientnet, eff_preprocess),
        ("InceptionV3",    build_inception,    unfreeze_inception,    inc_preprocess),
        ("ResNet50",       build_resnet50,     unfreeze_resnet50,     res_preprocess),
    ]:
        results[model_name] = train_transfer_model(
            model_name    = model_name,
            build_fn      = build_fn,
            unfreeze_fn   = unfreeze_fn,
            preprocess_fn = preprocess_fn,
            data_dir      = args.data_dir,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            save_dir      = save_dir,
            results_dir   = results_dir
        )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Validation Accuracy Summary")
    print("=" * 60)
    baseline_acc = results["Baseline CNN"]
    for name, acc in results.items():
        delta  = acc - baseline_acc
        marker = "← baseline" if name == "Baseline CNN" else \
                 f"({'+' if delta >= 0 else ''}{delta*100:.2f}% vs baseline)"
        print(f"  {name:<20}: {acc*100:.2f}%  {marker}")
    print("=" * 60)
    print("\n✅ All models trained. Weights saved to saved_models/")
    print("   Run python src/evaluate.py to generate evaluation reports.")


if __name__ == "__main__":
    main()