"""
utils/visualize.py
Visualization helpers: confidence bar chart, Grad-CAM heatmap overlay,
training history plots, and confusion matrix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


# ── Confidence Chart ──────────────────────────────────────────────────────────

def plot_confidence_chart(
    probabilities: np.ndarray,
    class_names: list,
    class_colors: dict,
) -> plt.Figure:
    """
    Horizontal bar chart of per-class probabilities.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = [class_colors.get(name, "#aaaaaa") for name in class_names]
    y_pos = np.arange(len(class_names))
    bars = ax.barh(y_pos, probabilities * 100, color=colors, edgecolor="none", height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9, color="white")
    ax.set_xlabel("Confidence (%)", color="white", fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", ha="left", fontsize=8, color="white",
        )

    plt.tight_layout()
    return fig


# ── Simulated Grad-CAM ────────────────────────────────────────────────────────

def generate_gradcam(
    image: np.ndarray,
    intensity: float = 0.6,
) -> plt.Figure:
    """
    Generates a simulated Grad-CAM-style heatmap overlay.
    In production, replace with a true Grad-CAM computation using
    tf.GradientTape on the last convolutional layer of EfficientNet-B0.

    Args:
        image:     RGB image, shape (224, 224, 3), dtype uint8.
        intensity: Blend factor for the heatmap overlay.

    Returns:
        matplotlib Figure with original and overlay side by side.
    """
    # Build a synthetic activation map centred on the image
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    activation = np.exp(-dist / (0.3 * min(h, w)))

    # Normalise and apply colormap
    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
    heatmap = np.uint8(255 * activation)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - intensity, heatmap_rgb, intensity, 0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.axis("off")

    axes[0].imshow(image)
    axes[0].set_title("Original", color="white", fontsize=10)
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay", color="white", fontsize=10)

    plt.tight_layout()
    return fig


# ── Training History ──────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None) -> plt.Figure:
    """
    Plot accuracy and loss curves from a Keras History object dict.

    Args:
        history:   dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        save_path: If provided, save figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history["accuracy"], label="Train Accuracy", color="#3498db")
    ax1.plot(history["val_accuracy"], label="Val Accuracy", color="#e74c3c", linestyle="--")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history["loss"], label="Train Loss", color="#2ecc71")
    ax2.plot(history["val_loss"], label="Val Loss", color="#e67e22", linestyle="--")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("EfficientNet-B0 Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm_matrix: np.ndarray,
    class_names: list,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot a labelled confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    short_names = [n.split(" ")[0] + " " + n.split(" ")[1] for n in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=8)

    thresh = cm_matrix.max() / 2.0
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            ax.text(
                j, i, str(cm_matrix[i, j]),
                ha="center", va="center",
                color="white" if cm_matrix[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
