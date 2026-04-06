"""
model_efficientnet.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Defines the EfficientNetB0 transfer learning model for the Lung Cancer
    Detection project.

    Preprocessing note:
        Preprocessing is handled by the data generator via the
        preprocessing_function parameter (efficientnet.preprocess_input),
        NOT inside the model. This makes the architecture simpler and
        fully compatible with save_weights_only / load_weights in TF 2.10.

    Training stages:
        Stage 1 — Feature extraction: frozen base, head only (5 epochs)
        Stage 2 — Fine-tuning: last 30 layers unfrozen (up to 25 epochs)

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam


def build_efficientnet(input_shape=(224, 224, 3), num_classes=3):
    """
    Builds EfficientNetB0 with a custom classification head.
    The pretrained base is fully frozen for Stage 1 training.

    Preprocessing is handled externally by the data generator —
    no Lambda or preprocessing layer inside this model.

    Args:
        input_shape (tuple): Shape of one input image. Default: (224, 224, 3).
        num_classes (int)  : Number of output classes. Default: 3.

    Returns:
        tf.keras.Model: Compiled model ready for Stage 1 training.
    """

    base_model = EfficientNetB0(
        weights     = "imagenet",
        include_top = False,
        input_shape = input_shape
    )
    base_model.trainable = False

    inputs  = layers.Input(shape=input_shape, name="input")
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="efficientnetb0")
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    trainable     = sum([len(l.trainable_weights)     for l in model.layers])
    non_trainable = sum([len(l.non_trainable_weights) for l in model.layers])
    print(f"  EfficientNetB0 built — Stage 1 (base frozen)")
    print(f"  Trainable weight tensors     : {trainable}")
    print(f"  Non-trainable weight tensors : {non_trainable}")

    return model


def unfreeze_efficientnet(model, fine_tune_layers=30):
    """
    Unfreezes the last N layers of EfficientNetB0 for Stage 2 fine-tuning.

    Args:
        model            (tf.keras.Model): Model from build_efficientnet().
        fine_tune_layers (int)           : Layers from the end to unfreeze.

    Returns:
        tf.keras.Model: Recompiled model ready for Stage 2.
    """
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer = Adam(learning_rate=1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    trainable = sum([len(l.trainable_weights) for l in model.layers])
    print(f"  EfficientNetB0 — Stage 2 (last {fine_tune_layers} layers unfrozen)")
    print(f"  Trainable weight tensors : {trainable}")
    return model