import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model.keras"
CLASS_PATH = BASE_DIR / "class_indices.json"

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Facial Emotion Recognition")
st.write("Upload a clear face photo to estimate the facial expression.")

EMOTION_INFO = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    if not CLASS_PATH.exists():
        return None
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

if model is None:
    st.error("Model file not found: emotion_model.keras")
    st.info("First prepare the training dataset and run: python train_model.py")
    st.stop()

if not class_names:
    st.error("class_indices.json is missing or empty.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        original = Image.open(uploaded_file).convert("RGB")
        st.image(original, caption="Uploaded image", use_container_width=True)

        if st.button("Predict emotion", type="primary"):
            # FER-style models use 48x48 grayscale images. The model Rescaling layer
            # converts pixel values from [0, 255] to [0, 1].
            face = original.convert("L").resize((48, 48))
            image_array = np.asarray(face, dtype=np.float32)
            image_array = image_array.reshape(1, 48, 48, 1)

            scores = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(scores))
            emotion = class_names[predicted_index]
            confidence = float(scores[predicted_index]) * 100.0

            emoji = EMOTION_INFO.get(emotion.lower(), "🙂")
            st.subheader("Prediction result")
            st.success(f"{emoji} Detected expression: **{emotion.upper()}**")
            st.metric("Model confidence", f"{confidence:.1f}%")
            st.progress(min(max(confidence / 100.0, 0.0), 1.0))

            st.caption(
                "This is an AI estimate of visible facial expression, "
                "not a reliable measurement of a person's actual feelings."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption("Capstone project demo • CNN-based facial expression classification")
