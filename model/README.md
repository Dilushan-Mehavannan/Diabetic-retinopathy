# model/

Place your trained model file here:

```
model/dr_efficientnet_b0.h5
```

## How to obtain the model

### Option A — Train yourself (recommended)
1. Open `notebooks/train_model.ipynb` in Google Colab (GPU runtime).
2. Follow the cells to download APTOS 2019, preprocess, and train.
3. The notebook saves `dr_efficientnet_b0.h5` to your Google Drive and copies it here.

### Option B — Use a pre-trained checkpoint
If you have a compatible EfficientNet-B0 checkpoint trained on APTOS 2019 (5-class softmax head),
rename it to `dr_efficientnet_b0.h5` and place it in this directory.

## Model architecture

| Component         | Detail                          |
|-------------------|---------------------------------|
| Base              | EfficientNet-B0 (ImageNet)      |
| Input shape       | (224, 224, 3)                   |
| Head              | GAP → BN → Dropout(0.3) → Dense(256) → Dropout(0.15) → Dense(5, softmax) |
| Output classes    | 5 (Grade 0 – Grade 4)           |
| Optimizer Phase 1 | Adam lr=1e-3 (frozen base)      |
| Optimizer Phase 2 | Adam lr=1e-5 (fine-tune last 20)|
| Loss              | Categorical cross-entropy       |

## App behaviour without the model

If `dr_efficientnet_b0.h5` is absent, the Streamlit app runs in **Demo Mode**:
predictions are randomly generated from the uploaded image's pixel statistics.
A warning banner is shown in the UI.
