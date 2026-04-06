"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Standalone evaluation script for the Lung Cancer Detection project.

    Loads all four trained models and evaluates them on the held-out test set.
    Generates and saves:
        - Classification report (precision, recall, F1 per class)
        - Confusion matrices
        - ROC curves
        - Final comparison table

Usage:
    From the project root:
        conda activate lung_cancer
        python src/evaluate.py

    Optional arguments:
        --data_dir  Path to dataset root  (default: data/lung_image_sets)

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
import itertools

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_preprocessing import get_data_generators, IMG_SIZE, BATCH_SIZE
from src.model_loader       import load_all_models, PREPROCESSING_FUNCTIONS

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
from tensorflow.keras.applications.resnet50     import preprocess_input as res_preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Lung Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]

# Maps model name → preprocessing function for its test generator
GENERATOR_PREPROCESSING = {
    "baseline_cnn"   : (True,  None),
    "efficientnetb0" : (False, eff_preprocess),
    "inceptionv3"    : (False, inc_preprocess),
    "resnet50"       : (False, res_preprocess),
}


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all lung cancer detection models on the test set."
    )
    parser.add_argument(
        "--data_dir",
        type    = str,
        default = "data/lung_image_sets",
        help    = "Path to dataset root folder (default: data/lung_image_sets)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, data_dir, rescale, preprocess_fn):
    """
    Generates predictions for the test set using the given model.

    Args:
        model         (tf.keras.Model): Loaded model.
        data_dir      (str)           : Path to dataset root.
        rescale       (bool)          : Whether to rescale pixels to [0,1].
        preprocess_fn (callable)      : Model-specific preprocessing function.

    Returns:
        tuple: (y_true, y_pred, y_prob)
    """
    _, _, test_gen = get_data_generators(
        data_dir               = data_dir,
        batch_size             = BATCH_SIZE,
        rescale                = rescale,
        preprocessing_function = preprocess_fn
    )

    test_gen.reset()
    y_prob = model.predict(
        test_gen,
        steps   = test_gen.n // BATCH_SIZE + 1,
        verbose = 1
    )
    y_prob = y_prob[:test_gen.n]
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array(test_gen.classes[:test_gen.n])

    return y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(all_preds, class_names, results_dir):
    """
    Plots and saves normalized confusion matrices for all four models.

    Args:
        all_preds   (dict)  : {model_name: (y_true, y_pred, y_prob)}
        class_names (list)  : Display names for each class.
        results_dir (str)   : Directory to save the plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    short     = ["Adeno.", "Normal", "Squamous"]

    for ax, (name, (y_true, y_pred, _)) in zip(axes, all_preds.items()):
        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        for i, j in itertools.product(range(3), range(3)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.suptitle("Confusion Matrices — Test Set",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved → {path}")


def plot_roc_curves(all_preds, class_names, results_dir):
    """
    Plots and saves ROC curves for all four models.

    Args:
        all_preds   (dict): {model_name: (y_true, y_pred, y_prob)}
        class_names (list): Display names for each class.
        results_dir (str) : Directory to save the plot.
    """
    colors      = ["#E8593C", "#1D9E75", "#7F77DD"]
    short_names = ["Adeno.", "Normal", "Squamous"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, (name, (y_true, _, y_prob)) in zip(axes, all_preds.items()):
        for i, (short, color) in enumerate(zip(short_names, colors)):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{short} (AUC={roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], color="#B4B2A9",
                linewidth=1, linestyle="--", label="Random")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("ROC Curves — Test Set",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main entry point — loads all models, generates predictions, and saves
    all evaluation plots and reports.
    """
    args        = parse_args()
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 62)
    print("  Lung Cancer Detection — Evaluation Pipeline")
    print("  Author: Ahmed Mohamed Abdelhady")
    print("=" * 62)

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading models...")
    all_models = load_all_models()

    # ── Generate predictions ──────────────────────────────────────────────
    all_preds = {}
    for model_name, model in all_models.items():
        print(f"\nGenerating predictions — {model_name} ...")
        rescale, preprocess_fn = GENERATOR_PREPROCESSING[model_name]
        y_true, y_pred, y_prob = get_predictions(
            model, args.data_dir, rescale, preprocess_fn
        )
        all_preds[model_name] = (y_true, y_pred, y_prob)

    # ── Classification reports ────────────────────────────────────────────
    print("\n")
    for name, (y_true, y_pred, _) in all_preds.items():
        print("=" * 62)
        print(f"  {name}")
        print("=" * 62)
        print(classification_report(
            y_true, y_pred,
            target_names = CLASS_NAMES,
            digits       = 4
        ))

    # ── Confusion matrices ────────────────────────────────────────────────
    plot_confusion_matrices(all_preds, CLASS_NAMES, results_dir)

    # ── ROC curves ────────────────────────────────────────────────────────
    plot_roc_curves(all_preds, CLASS_NAMES, results_dir)

    # ── Final comparison table ────────────────────────────────────────────
    print("=" * 65)
    print("  FINAL EVALUATION — TEST SET")
    print("=" * 65)
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'Mean AUC':>10}")
    print("-" * 65)

    best_acc = max(
        accuracy_score(y_true, y_pred)
        for y_true, y_pred, _ in all_preds.values()
    )

    for name, (y_true, y_pred, y_prob) in all_preds.items():
        acc      = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        aucs     = []
        for i in range(y_prob.shape[1]):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            aucs.append(auc(fpr, tpr))
        mean_auc = np.mean(aucs)
        marker   = "  ← best" if acc == best_acc else ""
        print(
            f"  {name:<20}"
            f" {acc*100:>9.2f}%"
            f" {f1_macro*100:>11.2f}%"
            f" {mean_auc:>10.4f}"
            f"{marker}"
        )

    print("=" * 65)
    print("\n✅ Evaluation complete. All plots saved to results/")


if __name__ == "__main__":
    main()