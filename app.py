import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import tempfile


#  PAGE CONFIGURATION

st.set_page_config(
    page_title="Aerial Object Classifier",
    layout="centered"
)

#  PATHS — UPDATE IF NEEDED

CNN_MODEL_PATH   = r"D:\Labmentix\aerial object\best_MobileNetV2_finetuned.keras"
YOLO_MODEL_PATH  = r"D:\Labmentix\aerial object\yolo_runs\bird_drone_detection2\weights\best.pt"

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["bird", "drone"]


#  LOAD MODELS (cached so they load only once)
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model(CNN_MODEL_PATH)

@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL_PATH)
    except Exception:
        return None


#  HELPER: CLASSIFY IMAGE
def classify_image(model, pil_image):
    """Preprocess image and run classification."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)         # (1, 224, 224, 3)
    prob = float(model.predict(arr, verbose=0)[0][0])
    # prob close to 0 → Bird, close to 1 → Drone
    label = "Drone" if prob >= 0.5 else "Bird"
    confidence = prob if label == "Drone" else 1 - prob
    return label, confidence


#  HELPER: YOLO DETECTION
def run_yolo_detection(yolo_model, pil_image):
    """Run YOLOv8 inference and return annotated image."""
    import cv2
    import numpy as np

    # Save temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    results = yolo_model.predict(
        source=tmp_path,
        conf=0.25,
        iou=0.45,
        verbose=False
    )
    os.unlink(tmp_path)

    # Get annotated image from result
    annotated = results[0].plot()   # numpy array BGR
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), results[0].boxes


#  UI — HEADER

st.title("Aerial Object Classifier")
st.markdown("### Bird vs Drone Detection using Deep Learning")
st.markdown(
    "Upload an aerial image to classify it as a **Bird** or **Drone**, "
    "and optionally detect objects with **YOLOv8** bounding boxes."
)
st.divider()


#  SIDEBAR — SETTINGS
with st.sidebar:
    st.header("Settings")
    use_yolo = st.toggle("Enable YOLOv8 Detection", value=False)
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Classifier: MobileNetV2")
    st.markdown("- Detector: YOLOv8n")
    st.markdown("- Classes: Bird, Drone")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- Train: 2662 images")
    st.markdown("- Val: 442 images")
    st.markdown("- Test: 215 images")


#  LOAD MODELS

with st.spinner("Loading classification model..."):
    try:
        clf_model = load_classification_model()
        st.success("Classification model loaded (MobileNetV2)")
    except Exception as e:
        st.error(f" Could not load classification model: {e}")
        st.stop()

yolo_model = None
if use_yolo:
    with st.spinner("Loading YOLOv8 model..."):
        yolo_model = load_yolo_model()
        if yolo_model:
            st.success("YOLOv8 model loaded")
        else:
            st.warning("YOLOv8 model not found. Train it first using yolov8_pipeline.py")

st.divider()


#  FILE UPLOADER

uploaded_file = st.file_uploader(
    "Upload an aerial image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image of a bird or drone"
)


#  PREDICTION
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    # ── Classification ───────────────────────
    with st.spinner("Classifying..."):
        label, confidence = classify_image(clf_model, pil_image)

    with col2:
        st.subheader("Classification Result")

        # Color based on class
        color = "#1e88e5" if label == "Bird" else "#e53935"
        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-top: 10px;
            ">
                <h1 style="color:{color}; margin:0">{label}</h1>
                <p style="font-size:18px; margin:8px 0">
                    Confidence: <strong>{confidence:.2%}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence bar
        st.markdown("#### Confidence Breakdown")
        bird_conf  = 1 - float(clf_model.predict(
            np.expand_dims(np.array(pil_image.convert("RGB").resize(IMG_SIZE)) / 255.0, 0),
            verbose=0)[0][0])
        drone_conf = 1 - bird_conf

        st.markdown("Bird")
        st.progress(float(bird_conf))
        st.caption(f"{bird_conf:.2%}")

        st.markdown("Drone")
        st.progress(float(drone_conf))
        st.caption(f"{drone_conf:.2%}")

    # ── YOLOv8 Detection ─────────────────────
    if use_yolo and yolo_model is not None:
        st.divider()
        st.subheader("YOLOv8 Object Detection")

        with st.spinner("Running YOLOv8 detection..."):
            annotated_img, boxes = run_yolo_detection(yolo_model, pil_image)

        st.image(annotated_img, caption="YOLOv8 Detection Result",
                 use_container_width=True)

        if boxes is not None and len(boxes) > 0:
            st.markdown(f"**Detected {len(boxes)} object(s):**")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = CLASS_NAMES[cls_id].capitalize() if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                st.markdown(f"- **{name}** — Confidence: `{conf:.2%}`")
        else:
            st.info("No objects detected above the confidence threshold (0.25).")

    st.divider()
    st.caption("Aerial Object Classification | Deep Learning Project | TensorFlow + YOLOv8")
