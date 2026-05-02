"""
model/train.py
Training pipeline for the EfficientNet-B0 DR classifier.
Run this script from the project root:
    python model/train.py --data_dir /path/to/aptos2019 --epochs 30
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_builder import build_model, unfreeze_model, get_callbacks
from utils.preprocess import preprocess_batch
from utils.visualize import plot_training_history, plot_confusion_matrix


# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 5
IMG_SIZE    = 224
BATCH_SIZE  = 32
SEED        = 42

CLASS_NAMES = [
    "No DR (Grade 0)", "Mild DR (Grade 1)", "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)", "Proliferative DR (Grade 4)",
]

# ── Data loader ───────────────────────────────────────────────────────────────

def load_aptos_dataset(data_dir: str, csv_path: str):
    """
    Load images and labels from the APTOS 2019 dataset structure:
        data_dir/
            train_images/   ← .png fundus images
            train.csv       ← columns: id_code, diagnosis

    Args:
        data_dir:  Root folder containing train_images/ and train.csv
        csv_path:  Path to the CSV (overrides default if provided)

    Returns:
        images (np.ndarray), labels (np.ndarray of int)
    """
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, "train_images", row["id_code"] + ".png")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(int(row["diagnosis"]))

    return np.array(images, dtype=object), np.array(labels)


def create_data_generators(
    images, labels, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2
):
    """
    Split data, preprocess, and create tf.data.Dataset pipelines with augmentation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_split, stratify=labels, random_state=SEED
    )

    # Preprocess
    X_train_proc = preprocess_batch(list(X_train), img_size)
    X_val_proc   = preprocess_batch(list(X_val),   img_size)

    # One-hot encode
    Y_train = to_categorical(y_train, NUM_CLASSES)
    Y_val   = to_categorical(y_val,   NUM_CLASSES)

    # Augmentation layer (train only)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def augment_fn(x, y):
        return augment(x, training=True), y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_proc, Y_train))
        .shuffle(len(X_train_proc), seed=SEED)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_proc, Y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, X_val_proc, y_val


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print(" Diabetic Retinopathy Classifier — Training")
    print("=" * 60)

    # GPU config
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✓ GPU detected: {gpus[0].name}")
    else:
        print("⚠ No GPU found, using CPU.")

    csv_path = args.csv_path or os.path.join(args.data_dir, "train.csv")
    print(f"\nLoading dataset from: {args.data_dir}")
    images, labels = load_aptos_dataset(args.data_dir, csv_path)
    print(f"  Total samples: {len(images)}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Grade {u}: {c} images")

    print("\nCreating data pipelines …")
    train_ds, val_ds, X_val, y_val = create_data_generators(
        images, labels, batch_size=args.batch_size
    )

    # Phase 1: Feature extraction (frozen base)
    print("\n[Phase 1] Feature extraction (frozen EfficientNet-B0) …")
    model = build_model(freeze_base=True)
    ckpt = os.path.join(args.output_dir, "dr_efficientnet_b0_phase1.h5")
    cb   = get_callbacks(ckpt, patience=5)

    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=min(args.epochs, 15), callbacks=cb, verbose=1,
    )

    # Phase 2: Fine-tuning (unfreeze last 20 layers)
    print("\n[Phase 2] Fine-tuning last 20 layers …")
    model   = unfreeze_model(model, unfreeze_from=-20)
    ckpt2   = os.path.join(args.output_dir, "dr_efficientnet_b0.h5")
    cb2     = get_callbacks(ckpt2, patience=7)

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=cb2, verbose=1,
        initial_epoch=len(history1.history["loss"]),
    )

    # Combine histories
    combined_history = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        combined_history[key] = (
            history1.history.get(key, []) + history2.history.get(key, [])
        )

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "dr_efficientnet_b0.h5")
    model.save(final_path)
    print(f"\n✓ Model saved to: {final_path}")

    # Plots
    fig_hist = plot_training_history(combined_history)
    fig_hist.savefig(os.path.join(args.output_dir, "training_history.png"), dpi=150)

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_val, y_pred)
    fig_cm = plot_confusion_matrix(cm, CLASS_NAMES)
    fig_cm.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))
    print("\n✓ Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR EfficientNet-B0 classifier")
    parser.add_argument("--data_dir",   required=True, help="Path to APTOS 2019 dataset root")
    parser.add_argument("--csv_path",   default=None,  help="Path to train.csv (optional)")
    parser.add_argument("--output_dir", default="model", help="Directory to save model & plots")
    parser.add_argument("--epochs",     type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(args)
