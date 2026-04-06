"""
data_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Centralized data loading, preprocessing, and augmentation pipeline for
    the Lung Cancer Detection project.

    This module is imported by train.py, evaluate.py, and predict.py.
    All preprocessing logic lives here — no duplication across scripts.

Split strategy:
    We use scikit-learn's train_test_split() on file paths to guarantee
    a clean, non-overlapping 70% / 15% / 15% split before any generator
    is built.

Main function:
    get_data_generators() → returns (train_gen, val_gen, test_gen)

Author : Ahmed
Project: Lung Cancer Detection using CNN + Transfer Learning
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.add_dll_directory(r"C:\Users\ahmed\anaconda3\envs\lung_cancer\Library\bin")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
RANDOM_SEED = 42
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]


def _build_dataframe(data_dir, classes):
    """
    Walks each class subfolder and returns a DataFrame with two columns:
        filepath — absolute path to the image file
        label    — class folder name (e.g. "lung_aca")

    Args:
        data_dir (str)  : Path to the dataset root folder.
        classes  (list) : List of class subfolder names.

    Returns:
        pd.DataFrame: Columns ["filepath", "label"]
    """
    filepaths, labels = [], []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                filepaths.append(os.path.join(folder, fname))
                labels.append(cls)
    return pd.DataFrame({"filepath": filepaths, "label": labels})


def get_data_generators(
    data_dir,
    img_size              = IMG_SIZE,
    batch_size            = BATCH_SIZE,
    val_split             = 0.15,
    test_split            = 0.15,
    rescale               = True,
    preprocessing_function = None,
):
    """
    Builds and returns three Keras ImageDataGenerators with a guaranteed
    non-overlapping 70% / 15% / 15% train / val / test split.

    Preprocessing strategy:
        - Baseline CNN      : rescale=True,  preprocessing_function=None
          → pixels normalized to [0, 1] by the generator

        - Transfer learning : rescale=False, preprocessing_function=<model fn>
          → raw [0, 255] pixels passed to the model-specific preprocessing
            function (e.g. efficientnet.preprocess_input) which applies the
            correct normalization for that architecture.

        This approach is more reliable than Lambda layers inside the model
        because standard ImageDataGenerator preprocessing is fully compatible
        with save_weights_only and load_weights in TF 2.10.

    Args:
        data_dir               (str)     : Path to dataset root.
        img_size               (tuple)   : Target (width, height).
        batch_size             (int)     : Images per batch.
        val_split              (float)   : Fraction for validation.
        test_split             (float)   : Fraction for testing.
        rescale                (bool)    : If True, normalize pixels to [0,1].
                                           Set False for transfer learning models.
        preprocessing_function (callable): Model-specific preprocessing function.
                                           Applied after loading, before the model.
                                           e.g. efficientnet.preprocess_input

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """

    rescale_factor = 1.0 / 255 if rescale else None

    df            = _build_dataframe(data_dir, CLASS_NAMES)
    temp_fraction = val_split + test_split

    train_df, temp_df = train_test_split(
        df,
        test_size    = temp_fraction,
        random_state = RANDOM_SEED,
        stratify     = df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size    = 0.50,
        random_state = RANDOM_SEED,
        stratify     = temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── TRAINING GENERATOR ────────────────────────────────────────────────
    # preprocessing_function is applied to every image after loading and
    # before augmentation. For transfer learning models, this replaces the
    # Lambda layer we previously included inside the model architecture.
    train_datagen = ImageDataGenerator(
        rescale                = rescale_factor,
        preprocessing_function = preprocessing_function,
        horizontal_flip        = True,
        vertical_flip          = True,
        rotation_range         = 20,
        zoom_range             = 0.10,
        brightness_range       = [0.90, 1.10]
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe   = train_df,
        x_col       = "filepath",
        y_col       = "label",
        target_size = img_size,
        batch_size  = batch_size,
        class_mode  = "categorical",
        shuffle     = True,
        seed        = RANDOM_SEED
    )

    # ── VALIDATION GENERATOR ──────────────────────────────────────────────
    val_datagen = ImageDataGenerator(
        rescale                = rescale_factor,
        preprocessing_function = preprocessing_function
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe   = val_df,
        x_col       = "filepath",
        y_col       = "label",
        target_size = img_size,
        batch_size  = batch_size,
        class_mode  = "categorical",
        shuffle     = False,
        seed        = RANDOM_SEED
    )

    # ── TEST GENERATOR ────────────────────────────────────────────────────
    test_datagen = ImageDataGenerator(
        rescale                = rescale_factor,
        preprocessing_function = preprocessing_function
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe   = test_df,
        x_col       = "filepath",
        y_col       = "label",
        target_size = img_size,
        batch_size  = batch_size,
        class_mode  = "categorical",
        shuffle     = False,
        seed        = RANDOM_SEED
    )

    print("─" * 55)
    print("  Data Generators Ready")
    print("─" * 55)
    print(f"  Image size             : {img_size[0]} × {img_size[1]} px")
    print(f"  Batch size             : {batch_size}")
    print(f"  Rescaling              : {'[0,1]' if rescale else 'None'}")
    print(f"  Preprocessing function : {preprocessing_function.__name__ if preprocessing_function else 'None'}")
    print(f"  Train samples          : {train_gen.n}")
    print(f"  Val   samples          : {val_gen.n}")
    print(f"  Test  samples          : {test_gen.n}")
    print(f"  Classes                : {train_gen.class_indices}")
    print("─" * 55)

    return train_gen, val_gen, test_gen