"""
model_resnet50.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Defines the ResNet50 transfer learning model for the Lung Cancer
    Detection project.

    Preprocessing note:
        Preprocessing is handled by the data generator via the
        preprocessing_function parameter (resnet50.preprocess_input),
        NOT inside the model.

Author : Ahmed Mohamed Abdelhady
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


def build_resnet50(input_shape=(224, 224, 3), num_classes=3):
    """
    Builds ResNet50 with a custom classification head.
    The pretrained base is fully frozen for Stage 1 training.

    Args:
        input_shape (tuple): Shape of one input image. Default: (224, 224, 3).
        num_classes (int)  : Number of output classes. Default: 3.

    Returns:
        tf.keras.Model: Compiled model ready for Stage 1 training.
    """

    base_model = ResNet50(
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

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50")
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    trainable     = sum([len(l.trainable_weights)     for l in model.layers])
    non_trainable = sum([len(l.non_trainable_weights) for l in model.layers])
    print(f"  ResNet50 built — Stage 1 (base frozen)")
    print(f"  Trainable weight tensors     : {trainable}")
    print(f"  Non-trainable weight tensors : {non_trainable}")

    return model


def unfreeze_resnet50(model, fine_tune_layers=30):
    """
    Unfreezes the last N layers of ResNet50 for Stage 2 fine-tuning.

    Args:
        model            (tf.keras.Model): Model from build_resnet50().
        fine_tune_layers (int)           : Layers from the end to unfreeze.

    Returns:
        tf.keras.Model: Recompiled model ready for Stage 2.
    """
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer = Adam(learning_rate=1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    trainable = sum([len(l.trainable_weights) for l in model.layers])
    print(f"  ResNet50 — Stage 2 (last {fine_tune_layers} layers unfrozen)")
    print(f"  Trainable weight tensors : {trainable}")
    return model