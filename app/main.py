import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import preprocess_image
from utils.visualize import plot_confidence_chart, generate_gradcam

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "No DR (Grade 0)",
    "Mild DR (Grade 1)",
    "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)",
    "Proliferative DR (Grade 4)",
]

CLASS_DESCRIPTIONS = {
    "No DR (Grade 0)": "No signs of diabetic retinopathy detected. Regular annual screening is still recommended.",
    "Mild DR (Grade 1)": "Mild nonproliferative diabetic retinopathy. Microaneurysms are present. Follow-up in 12 months.",
    "Moderate DR (Grade 2)": "Moderate nonproliferative DR. More microaneurysms and some vascular changes. Follow-up in 6 months.",
    "Severe DR (Grade 3)": "Severe nonproliferative DR. Significant vascular changes. Referral to ophthalmologist recommended.",
    "Proliferative DR (Grade 4)": "Proliferative DR — most advanced stage. New blood vessels forming. Urgent ophthalmologist referral required.",
}

CLASS_COLORS = {
    "No DR (Grade 0)": "#2ecc71",
    "Mild DR (Grade 1)": "#f1c40f",
    "Moderate DR (Grade 2)": "#e67e22",
    "Severe DR (Grade 3)": "#e74c3c",
    "Proliferative DR (Grade 4)": "#8e44ad",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "dr_efficientnet_b0.h5")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/eye.png", width=80)
    st.title("DR Classifier")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This tool uses **EfficientNet-B0** trained on retinal fundus images to "
        "classify **Diabetic Retinopathy (DR)** across 5 severity grades (0–4)."
    )
    st.markdown("### Grading Scale")
    for name, color in CLASS_COLORS.items():
        st.markdown(f"<span style='color:{color}'>■</span> {name}", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("⚠️ For research/educational use only. Not a medical device.")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("👁️ Diabetic Retinopathy Prediction & Classification")
st.markdown(
    "Upload a **retinal fundus image** to detect and grade diabetic retinopathy "
    "using a deep learning model built on **EfficientNet-B0**."
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "Choose a fundus image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

        analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    else:
        st.info("Please upload a retinal fundus image to begin analysis.")
        analyze_btn = False

with col2:
    st.subheader("📊 Prediction Results")

    if uploaded_file is not None and analyze_btn:
        with st.spinner("Preprocessing image and running inference..."):
            img_array = np.array(image)
            processed = preprocess_image(img_array)

            model = load_model()

            if model is not None:
                preds = model.predict(processed, verbose=0)[0]
            else:
                # Demo mode: simulate predictions when model file not present
                np.random.seed(int(img_array.mean()) % 100)
                raw = np.random.dirichlet(np.ones(5) * 2)
                preds = raw

            pred_class_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = float(preds[pred_class_idx]) * 100

        # Result card
        color = CLASS_COLORS[pred_class]
        st.markdown(
            f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        padding:16px; border-radius:8px; margin-bottom:16px;">
                <h3 style="color:{color}; margin:0;">Prediction: {pred_class}</h3>
                <p style="margin:4px 0 0;">Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Clinical Note:** {CLASS_DESCRIPTIONS[pred_class]}")

        st.markdown("#### Class Probability Distribution")
        fig = plot_confidence_chart(preds, CLASS_NAMES, CLASS_COLORS)
        st.pyplot(fig, use_container_width=True)

        # Grad-CAM
        st.markdown("#### 🌡️ Grad-CAM Heatmap (Attention Map)")
        img_array_resized = cv2.resize(np.array(image), (224, 224))
        gradcam_fig = generate_gradcam(img_array_resized)
        st.pyplot(gradcam_fig, use_container_width=True)
        st.caption("Heatmap highlights regions the model focuses on for prediction.")

        if model is None:
            st.warning(
                "⚠️ **Demo Mode**: No trained model found at `model/dr_efficientnet_b0.h5`. "
                "Predictions are simulated. Train the model using `notebooks/train_model.ipynb` "
                "and place the `.h5` file in the `model/` directory."
            )
    elif uploaded_file is not None and not analyze_btn:
        st.info("Click **Analyze Image** to run prediction.")
    else:
        st.empty()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "Diabetic Retinopathy Classifier | EfficientNet-B0 | TensorFlow/Keras | Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
