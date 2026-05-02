"""
model/model_builder.py
Builds the EfficientNet-B0 transfer-learning model for Diabetic Retinopathy
classification (5 classes: Grade 0–4).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
import os


NUM_CLASSES = 5
IMG_SIZE    = 224
BATCH_SIZE  = 32


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: int    = IMG_SIZE,
    dropout_rate: float = 0.3,
    freeze_base: bool   = True,
) -> tf.keras.Model:
    """
    Construct the EfficientNet-B0 transfer-learning model.

    Architecture:
        EfficientNet-B0 (ImageNet weights, optionally frozen)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dropout(dropout_rate)
        → Dense(256, relu)
        → Dropout(dropout_rate / 2)
        → Dense(num_classes, softmax)

    Args:
        num_classes:   Number of output classes (5 for DR grading).
        img_size:      Spatial input size (224 for EfficientNet-B0).
        dropout_rate:  Dropout rate for regularisation.
        freeze_base:   If True, freeze base model weights (feature extraction mode).

    Returns:
        Compiled tf.keras.Model.
    """
    # Base model
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = not freeze_base

    # Classification head
    inputs  = tf.keras.Input(shape=(img_size, img_size, 3))
    x       = base_model(inputs, training=not freeze_base)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(dropout_rate)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="DR_EfficientNetB0")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def unfreeze_model(model: tf.keras.Model, unfreeze_from: int = -20) -> tf.keras.Model:
    """
    Unfreeze the last |unfreeze_from| layers of the base model for fine-tuning.
    Recompiles with a lower learning rate.

    Args:
        model:          The compiled model returned by build_model().
        unfreeze_from:  Negative index; layers after this index are unfrozen.

    Returns:
        Recompiled model.
    """
    base_model = model.layers[1]  # EfficientNetB0 layer
    base_model.trainable = True

    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_callbacks(checkpoint_path: str, patience: int = 5) -> list:
    """
    Return a standard set of Keras callbacks for training.

    Includes:
        - ModelCheckpoint (saves best weights)
        - EarlyStopping
        - ReduceLROnPlateau
        - TensorBoard (optional)
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    cb = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return cb


def model_summary(model: tf.keras.Model) -> None:
    """Print a formatted model summary."""
    model.summary(line_length=100)
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    total         = trainable + non_trainable
    print(f"\nTotal params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Non-trainable    : {non_trainable:,}")
