"""
model_baseline.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Defines the baseline CNN architecture for the Lung Cancer Detection project.

    This is a custom CNN built entirely from scratch — no pretrained weights.
    It serves as the benchmark model that all transfer learning models must beat.

Architecture summary:
    4 × Convolutional Blocks (Conv2D → BatchNorm → ReLU → MaxPool)
    GlobalAveragePooling2D
    Dense(128) → Dropout(0.5) → Dense(3, softmax)

Usage:
    from src.model_baseline import build_baseline_model
    model = build_baseline_model(input_shape=(224, 224, 3), num_classes=3)
    model.summary()

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

from tensorflow.keras import layers, models # type: ignore


def build_baseline_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Builds and returns the baseline CNN model.

    Args:
        input_shape (tuple): Shape of one input image (height, width, channels).
                             Default: (224, 224, 3) — RGB images at 224×224.
        num_classes (int)  : Number of output classes.
                             Default: 3 — lung_aca, lung_n, lung_scc.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """

    # ── Input layer ──────────────────────────────────────────────────────────
    # Defines the shape of one image the model expects
    inputs = layers.Input(shape=input_shape, name="input")

    # ── Helper: single convolutional block ───────────────────────────────────
    # We define this as an inner function to avoid repeating the same 4 lines
    # four times. Each block: Conv2D → BatchNorm → ReLU → MaxPool
    def conv_block(x, filters, block_name):
        """
        One convolutional block: feature extraction + normalization + downsampling.

        Args:
            x          : Input tensor from the previous layer.
            filters    : Number of convolutional filters (controls depth of feature maps).
            block_name : Name prefix for Keras layer naming (helps with debugging).

        Returns:
            Output tensor after Conv → BN → ReLU → MaxPool.
        """
        # Conv2D: slides a 3×3 filter over the image to detect local patterns.
        # padding="same" keeps the spatial dimensions the same after convolution.
        x = layers.Conv2D(
            filters,
            kernel_size = (3, 3),
            padding     = "same",
            use_bias    = False,      # BatchNorm handles the bias term — no need to duplicate
            name        = f"{block_name}_conv"
        )(x)

        # BatchNormalization: normalizes the output of each layer so values
        # don't grow too large or too small — makes training faster and stabler.
        x = layers.BatchNormalization(name=f"{block_name}_bn")(x)

        # ReLU activation: replaces all negative values with 0.
        # This introduces non-linearity — without it, stacking layers
        # would just be the same as one linear transformation.
        x = layers.Activation("relu", name=f"{block_name}_relu")(x)

        # MaxPooling: takes the maximum value in each 2×2 region.
        # This halves the spatial dimensions (e.g. 112×112 → 56×56),
        # reducing computation and keeping only the strongest activations.
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)

        return x

    # ── Convolutional blocks ──────────────────────────────────────────────────
    # Filters double with each block — early layers detect simple features
    # (edges, corners), deeper layers detect complex patterns (textures, shapes).
    x = conv_block(inputs, filters=32,  block_name="block1")  # 224→112
    x = conv_block(x,      filters=64,  block_name="block2")  # 112→56
    x = conv_block(x,      filters=128, block_name="block3")  # 56→28
    x = conv_block(x,      filters=256, block_name="block4")  # 28→14

    # ── Global Average Pooling ────────────────────────────────────────────────
    # Compresses each of the 256 feature maps (14×14) into a single number
    # by taking the average. Output: a vector of 256 values.
    # Much more efficient than Flatten() which would give 14×14×256 = 50,176 values.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ── Fully connected head ──────────────────────────────────────────────────
    # Dense(128): learns high-level combinations of the extracted features
    x = layers.Dense(128, activation="relu", name="dense_128")(x)

    # Dropout(0.5): randomly disables 50% of neurons during each training step.
    # Forces the network not to over-rely on specific neurons → reduces overfitting.
    # Dropout is disabled automatically during inference (prediction).
    x = layers.Dropout(0.5, name="dropout")(x)

    # Output layer: 3 neurons (one per class), softmax converts raw scores
    # into probabilities that sum to 1.0.
    # e.g. [0.02, 0.95, 0.03] → 95% confident this is lung_n (Normal)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # ── Assemble and compile the model ───────────────────────────────────────
    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

    model.compile(
        # Adam: adaptive learning rate optimizer — adjusts the learning rate
        # for each parameter individually. Standard choice for CNNs.
        optimizer = "adam",

        # Categorical crossentropy: standard loss for multi-class classification
        # when labels are one-hot encoded (e.g. [0, 1, 0]).
        loss = "categorical_crossentropy",

        # Track accuracy during training so we can plot learning curves
        metrics = ["accuracy"]
    )

    return model 