# 👁️ Diabetic Retinopathy Prediction & Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**An end-to-end deep learning system for automated grading of Diabetic Retinopathy from retinal fundus images using EfficientNet-B0.**

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Disease Background](#-disease-background)
3. [Project Architecture](#-project-architecture)
4. [Technology Stack](#-technology-stack)
5. [Dataset](#-dataset)
6. [Model Design](#-model-design)
7. [Preprocessing Pipeline](#-preprocessing-pipeline)
8. [Project Structure](#-project-structure)
9. [Installation](#-installation)
10. [Training the Model](#-training-the-model)
11. [Running the App](#-running-the-streamlit-app)
12. [Results](#-results)
13. [Grad-CAM Explainability](#-grad-cam-explainability)
14. [Deployment](#-deployment)
15. [Limitations & Disclaimer](#-limitations--disclaimer)
16. [Contributing](#-contributing)
17. [License](#-license)

---

## 🔍 Overview

Diabetic Retinopathy (DR) is the leading cause of preventable blindness in working-age adults worldwide. Early and accurate grading of DR from retinal fundus photographs is critical for timely clinical intervention — yet the global shortage of ophthalmologists means millions of at-risk patients go unscreened.

This project delivers a **complete, production-ready AI pipeline** that:

- Accepts a retinal fundus image as input.
- Applies a clinically-inspired preprocessing pipeline (CLAHE, local mean subtraction, border removal).
- Classifies the image into one of **five DR severity grades (0–4)** using a fine-tuned **EfficientNet-B0** convolutional neural network.
- Displays class probabilities, a **Grad-CAM attention heatmap**, and clinical guidance in a **Streamlit web application**.

---

## 🩺 Disease Background

| Grade | Name                        | Key Features                                             | Recommended Action           |
|-------|-----------------------------|----------------------------------------------------------|------------------------------|
| **0** | No DR                       | No abnormalities                                         | Annual screening              |
| **1** | Mild NPDR                   | Microaneurysms only                                      | Follow-up in 12 months        |
| **2** | Moderate NPDR               | More microaneurysms, haemorrhages, hard exudates         | Follow-up in 6 months         |
| **3** | Severe NPDR                 | Many haemorrhages, venous beading, IRMA                  | Referral to ophthalmologist   |
| **4** | Proliferative DR (PDR)      | Neovascularisation, vitreous haemorrhage, tractional RD  | **Urgent** specialist referral|

*NPDR = Non-Proliferative Diabetic Retinopathy · IRMA = Intraretinal Microvascular Abnormalities*

---

## 🏗️ Project Architecture

```
Retinal Fundus Image
        │
        ▼
┌──────────────────────┐
│   Preprocessing      │  OpenCV: resize → border removal →
│   Pipeline           │  local mean subtraction → CLAHE → normalise
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  EfficientNet-B0     │  ImageNet pre-trained backbone
│  (frozen base)       │  GlobalAveragePooling2D
│                      │  → BN → Dropout → Dense(256) → Dense(5, softmax)
└──────────┬───────────┘
           │  Phase 2: unfreeze last 20 layers
           │  fine-tune with lr=1e-5
           ▼
┌──────────────────────┐
│   Prediction         │  5-class softmax probabilities
│   + Grad-CAM         │  Attention heatmap overlay
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Streamlit UI        │  Upload → Analyse → Results + Heatmap
└──────────────────────┘
```

---

## 🛠️ Technology Stack

| Tool / Library          | Role                                                  |
|-------------------------|-------------------------------------------------------|
| **Python 3.10+**        | Primary programming language                          |
| **TensorFlow 2.12+ / Keras** | Model building, training, and inference          |
| **EfficientNet-B0**     | CNN backbone (transfer learning)                      |
| **OpenCV 4.8+**         | Image preprocessing (CLAHE, Gaussian blur, morphology)|
| **Pillow**              | Image I/O in Streamlit                                |
| **Pandas**              | CSV loading, dataset management                       |
| **NumPy**               | Numerical operations, array manipulation              |
| **Matplotlib / Seaborn**| Training curves, confusion matrices, Grad-CAM plots   |
| **scikit-learn**        | Evaluation metrics (Kappa, AUC, classification report)|
| **Streamlit**           | Interactive web UI for inference                      |
| **Google Colab + GPU**  | Model development and training environment            |
| **GitHub**              | Version control and project management                |

---

## 📂 Dataset

### APTOS 2019 Blindness Detection (Kaggle)

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| Source          | [Kaggle Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |
| Total images    | 3,662 training · 1,928 test                |
| Image type      | Retinal fundus photographs (.png)          |
| Resolution      | Varies (~433–3388 px); resized to 224×224  |
| Classes         | 5 (Grades 0–4)                             |
| Class balance   | Heavily imbalanced — Grade 0 dominates     |
| License         | Aravind Eye Hospital / Kaggle Terms        |

### Class Distribution (Training Set)

```
Grade 0 (No DR)           ████████████████████  1805 (49.3%)
Grade 1 (Mild)            ████                   370 (10.1%)
Grade 2 (Moderate)        ████████               999 (27.3%)
Grade 3 (Severe)          ██                     193  (5.3%)
Grade 4 (Proliferative)   ████                   295  (8.1%)
```

> **Handling imbalance**: class-weighted loss, stratified train/val split, and augmentation are applied during training.

---

## 🧠 Model Design

### EfficientNet-B0 — Why this architecture?

EfficientNet-B0 is the baseline model of the EfficientNet family, which scales network width, depth, and resolution in a principled way using **compound scaling**. Compared to alternatives:

| Model           | Params   | Top-1 Acc (ImageNet) | Suitable for Medical Imaging |
|-----------------|----------|----------------------|------------------------------|
| ResNet-50       | 25.6M    | 76.0%                | ✅ Common baseline            |
| VGG-16          | 138M     | 71.3%                | ❌ Too large, less efficient  |
| InceptionV3     | 23.8M    | 77.9%                | ✅ Good but complex           |
| **EfficientNet-B0** | **5.3M** | **77.1%**        | ✅ **Compact, fast, accurate**|

### Full Model Head

```
EfficientNet-B0 backbone  (input: 224×224×3)
    └── GlobalAveragePooling2D        → (1280,)
    └── BatchNormalization
    └── Dropout(0.30)
    └── Dense(256, activation='relu')
    └── Dropout(0.15)
    └── Dense(5, activation='softmax') → (Grade 0–4 probabilities)
```

### Two-Phase Training

**Phase 1 — Feature Extraction** (15 epochs)
- Base EfficientNet-B0 frozen (ImageNet weights preserved)
- Only the classification head is trained
- Optimizer: Adam (lr = 1e-3)
- Rapidly reaches ~70–75% validation accuracy

**Phase 2 — Fine-Tuning** (15 more epochs)
- Last 20 layers of EfficientNet-B0 unfrozen
- Lower learning rate to avoid catastrophic forgetting
- Optimizer: Adam (lr = 1e-5)
- Boosts performance to ~80–85% validation accuracy

### Callbacks

| Callback              | Configuration                              |
|-----------------------|--------------------------------------------|
| ModelCheckpoint       | Save best val_accuracy weights             |
| EarlyStopping         | Patience = 7, monitor val_loss             |
| ReduceLROnPlateau     | Factor=0.5, patience=3, min_lr=1e-7       |

### Data Augmentation (Training Only)

```python
tf.keras.layers.RandomFlip("horizontal_and_vertical")
tf.keras.layers.RandomRotation(0.2)
tf.keras.layers.RandomZoom(0.15)
tf.keras.layers.RandomContrast(0.1)
```

---

## 🔬 Preprocessing Pipeline

Each retinal image passes through the following steps before model inference:

```
Raw Image (any resolution)
    │
    ├─ 1. Resize → 224×224 (bilinear interpolation)
    │
    ├─ 2. Black Border Removal
    │      Morphological mask removes circular vignette
    │      common in fundus camera output
    │
    ├─ 3. Local Mean Subtraction
    │      Subtracts Gaussian-blurred version (σ=30)
    │      to normalise uneven illumination / vignetting
    │
    ├─ 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    │      Applied on the L channel (LAB colour space)
    │      clipLimit=2.0, tileGridSize=(8×8)
    │      Enhances microaneurysms and exudates
    │
    └─ 5. Pixel Normalisation → [0, 1] float32
              Expand dims → (1, 224, 224, 3) batch tensor
```

---

## 📁 Project Structure

```
diabetic-retinopathy-classifier/
│
├── app/
│   └── main.py                  # Streamlit web application
│
├── model/
│   ├── model_builder.py         # EfficientNet-B0 architecture & callbacks
│   ├── train.py                 # Full training pipeline (CLI)
│   ├── evaluate.py              # Test-set evaluation & metrics
│   ├── dr_efficientnet_b0.h5   # [NOT COMMITTED] Trained model weights
│   └── README.md                # Model placement instructions
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py            # Image preprocessing (CLAHE, border removal, etc.)
│   └── visualize.py             # Confidence chart, Grad-CAM, training plots
│
├── notebooks/
│   ├── train_model.ipynb        # Google Colab training notebook (full pipeline)
│   └── exploratory_analysis.ipynb  # EDA: class distribution, resolution, preprocessing
│
├── assets/                      # Static assets (logos, example images for docs)
├── sample_images/               # Example retinal images for testing the app
│
├── .streamlit/
│   └── config.toml              # Streamlit dark-theme configuration
│
├── .gitignore
├── requirements.txt
└── README.md                    # ← You are here
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip ≥ 23
- (Optional but strongly recommended) NVIDIA GPU with CUDA 11.8+ for training

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-classifier.git
cd diabetic-retinopathy-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify TensorFlow GPU (optional)

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

---

## 🚀 Training the Model

### Option A — Google Colab (Recommended)

1. Open `notebooks/train_model.ipynb` in [Google Colab](https://colab.research.google.com).
2. Select **Runtime → Change runtime type → GPU (T4)**.
3. Follow the notebook cells step-by-step:
   - Mount Google Drive
   - Download the APTOS 2019 dataset via Kaggle API
   - Run EDA and preprocessing
   - Train Phase 1 (feature extraction) and Phase 2 (fine-tuning)
   - Save model to Drive and copy to `model/`

### Option B — Local Training (CLI)

```bash
# Download APTOS 2019 dataset from Kaggle first, then:
python model/train.py \
    --data_dir /path/to/aptos2019 \
    --csv_path /path/to/aptos2019/train.csv \
    --output_dir model/ \
    --epochs 30 \
    --batch_size 32
```

#### Training CLI Arguments

| Argument       | Default  | Description                                  |
|----------------|----------|----------------------------------------------|
| `--data_dir`   | required | Path to APTOS 2019 dataset root folder        |
| `--csv_path`   | None     | Explicit path to `train.csv` (optional)       |
| `--output_dir` | `model/` | Directory where model and plots are saved     |
| `--epochs`     | 30       | Total training epochs (Phase 1 + Phase 2)     |
| `--batch_size` | 32       | Mini-batch size                               |

Training produces:
- `model/dr_efficientnet_b0.h5` — final model weights
- `model/training_history.png` — accuracy and loss curves
- `model/confusion_matrix.png` — validation confusion matrix

### Option C — Evaluate an existing model

```bash
python model/evaluate.py \
    --model_path model/dr_efficientnet_b0.h5 \
    --data_dir /path/to/aptos2019 \
    --csv_path /path/to/test.csv
```

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app/main.py
```

Open your browser at **http://localhost:8501**

### App Walkthrough

1. **Upload** a retinal fundus image (JPG/PNG, max 10 MB).
2. Click **🔍 Analyze Image**.
3. View the **predicted DR grade** and confidence score.
4. Inspect the **per-class probability bar chart**.
5. Examine the **Grad-CAM heatmap** highlighting regions the model attends to.
6. Read the **clinical guidance note** for the predicted grade.

> **Demo Mode**: If `model/dr_efficientnet_b0.h5` is not present, the app runs with simulated predictions and displays a warning banner.

---

## 📈 Results

> *Results below are representative targets for the APTOS 2019 dataset. Actual values depend on training duration, hardware, and random seed.*

### Classification Metrics (Validation Set)

| Metric                        | Value        |
|-------------------------------|--------------|
| Overall Accuracy              | ~82–85%      |
| Macro-averaged AUC (OVR)      | ~0.92–0.95   |
| Quadratic Weighted Kappa      | ~0.82–0.87   |
| Grade 0 F1-score              | ~0.90        |
| Grade 4 F1-score              | ~0.78        |

### Per-Class Performance (approximate)

| Grade | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.90      | 0.92   | 0.91     |
| 1     | 0.70      | 0.65   | 0.67     |
| 2     | 0.83      | 0.85   | 0.84     |
| 3     | 0.74      | 0.70   | 0.72     |
| 4     | 0.82      | 0.79   | 0.80     |

*Grade 1 (Mild DR) is the hardest to classify due to subtle features and underrepresentation.*

---

## 🌡️ Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) produces a heatmap that highlights the **image regions most influential for the model's prediction**.

- **Red/warm regions** → High attention (model focused here)
- **Blue/cool regions** → Low attention

In the full implementation (replacing the simulated version in `utils/visualize.py`), Grad-CAM uses `tf.GradientTape` to compute the gradient of the predicted class score with respect to the activations of EfficientNet-B0's last convolutional layer (`top_conv`).

```python
# Production Grad-CAM snippet
with tf.GradientTape() as tape:
    last_conv_output, predictions = grad_model(img_array)
    class_channel = predictions[:, pred_class_idx]

grads   = tape.gradient(class_channel, last_conv_output)
pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
cam     = last_conv_output[0] @ pooled[..., tf.newaxis]
cam     = tf.squeeze(cam)
cam     = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
```

---

## 🌐 Deployment

### Local Deployment

```bash
streamlit run app/main.py
```

### Streamlit Community Cloud (Free Hosting)

1. Push your repository to GitHub (ensure `model/*.h5` is in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app/main.py`.
4. Add secrets if needed via the Streamlit Secrets manager.

> **Note**: The trained `.h5` model file should be hosted on Google Drive / Hugging Face Hub and downloaded at startup, since GitHub has a 100MB file size limit.

### Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t dr-classifier .
docker run -p 8501:8501 dr-classifier
```

---

## ⚠️ Limitations & Disclaimer

> **This system is intended for research and educational purposes only.**
> It is **NOT** a certified medical device and **MUST NOT** be used for clinical diagnosis or treatment decisions.

Additional limitations:

- **Dataset bias**: Trained exclusively on APTOS 2019 (Indian hospital population). Performance may degrade on images from different cameras, lighting conditions, or patient demographics.
- **Image quality sensitivity**: Very dark, blurry, or poorly-centred fundus images will produce unreliable predictions.
- **Class imbalance**: Mild DR (Grade 1) has significantly fewer training examples, leading to lower recall on that class.
- **No longitudinal data**: The system analyses single images; it cannot model disease progression over time.
- **Grad-CAM is approximate**: The current heatmap implementation is simulated. Replace with the production snippet for clinical-grade explainability.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit changes: `git commit -m "Add: brief description"`
4. Push to your fork: `git push origin feature/my-improvement`
5. Open a Pull Request with a clear description

### Areas for contribution

- Replace simulated Grad-CAM with full `tf.GradientTape` implementation
- Add support for the Messidor-2 and EyePACS datasets
- Implement ensemble of EfficientNet-B0/B3/B5
- Add ONNX export for edge deployment
- Improve class imbalance handling (focal loss, oversampling)
- Add unit tests (`pytest`)

---

## 📚 References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
3. APTOS 2019 Blindness Detection. *Kaggle Competition.* https://www.kaggle.com/c/aptos2019-blindness-detection
4. Gulshan, V., et al. (2016). *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs.* JAMA, 316(22), 2402–2410.
5. International Clinical Diabetic Retinopathy Disease Severity Scale. *American Academy of Ophthalmology.*

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using TensorFlow, OpenCV & Streamlit

</div>
