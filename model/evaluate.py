"""
model/evaluate.py
Evaluate a saved DR model on a test set and generate performance reports.

Usage:
    python model/evaluate.py \
        --model_path model/dr_efficientnet_b0.h5 \
        --data_dir /path/to/aptos2019 \
        --csv_path /path/to/test.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import preprocess_batch
from utils.visualize import plot_confusion_matrix

CLASS_NAMES = [
    "No DR (Grade 0)", "Mild DR (Grade 1)", "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)", "Proliferative DR (Grade 4)",
]


def evaluate(args):
    print("Loading model …")
    model = tf.keras.models.load_model(args.model_path)
    print(f"  ✓ Loaded from {args.model_path}")

    df = pd.read_csv(args.csv_path)
    images, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(args.data_dir, "test_images", row["id_code"] + ".png")
        if not os.path.exists(path):
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(int(row["diagnosis"]))

    print(f"  ✓ Loaded {len(images)} test images")
    X = preprocess_batch(images)
    y_true = np.array(labels)

    preds = model.predict(X, batch_size=32, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

    try:
        from tensorflow.keras.utils import to_categorical
        y_onehot = to_categorical(y_true, 5)
        auc = roc_auc_score(y_onehot, preds, multi_class="ovr", average="macro")
        print(f"Macro-averaged AUC:       {auc:.4f}")
    except Exception as e:
        print(f"AUC computation skipped: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plot_confusion_matrix(cm, CLASS_NAMES)
    out_path = os.path.join(os.path.dirname(args.model_path), "test_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n✓ Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--csv_path",   required=True)
    args = parser.parse_args()
    evaluate(args)
