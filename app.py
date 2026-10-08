from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from xai import GradCAMError, generate_detection_gradcam
from xai.visualization import blend_cam, colorize_cam, crop_detection_context, draw_detection_box

MODEL_PATH = Path("Colour.pt")
st.set_page_config(layout="wide", page_title="PCB Defect Detection Engine", page_icon="⚙️")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Inter:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;font-weight:700!important}.stButton>button{border:2px solid #76b900!important;background:linear-gradient(135deg,rgba(85,133,0,.8),rgba(118,185,0,.2))!important;color:#fff!important;border-radius:2px!important;font-weight:700!important;text-transform:uppercase}.stButton>button:hover{border-color:#94da32!important;box-shadow:0 0 10px rgba(118,185,0,.5)}div[data-testid="stMetric"]{background:#1a1a1a;padding:1rem;border-left:2px solid #76b900;border-radius:2px}thead tr th{background:#1a1a1a!important;color:#76b900!important}tbody tr td{background:transparent!important}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model file '{path}' was not found.")
    return YOLO(path)


def inspect(model: YOLO, image_rgb: np.ndarray) -> dict[str, object]:
    result = model.predict(image_rgb, verbose=False)[0]
    detections: list[dict[str, object]] = []
    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i].item())
        detections.append({"id": f"DEF-{i + 1:03d}", "class_id": class_id,
            "class_name": result.names[class_id], "confidence": float(result.boxes.conf[i].item()),
            "xyxy": result.boxes.xyxy[i].detach().cpu().numpy().tolist()})
    return {"image_rgb": image_rgb, "annotated": result.plot(), "detections": detections}


def density_heatmap(image: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    """Existing box-derived density graphic, intentionally not an XAI result."""
    layer = np.zeros(image.shape[:2], np.float32)
    for item in detections:
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cv2.circle(layer, ((x1+x2)//2, (y1+y2)//2), max(30, min((x2-x1)//2, (y2-y1)//2)*2), 1., -1)
    layer = cv2.GaussianBlur(layer, (151, 151), 0)
    if layer.max(): layer /= layer.max()
    heat = cv2.applyColorMap(np.uint8(layer * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), .5, heat, .5, 0), cv2.COLOR_BGR2RGB)


def display_xai(model: YOLO, record: dict[str, object]) -> None:
    detections = record["detections"]
    if not detections: return
    st.markdown("---")
    st.markdown("### 07 / Explainable AI — Grad-CAM")
    st.caption("Model-based post-hoc explanation: warmer/brighter regions contributed more strongly to this selected prediction. It is not proof that the prediction is correct.")
    selected = st.selectbox("Selected Detection", [d["id"] for d in detections], key="xai_detection")
    detection = next(d for d in detections if d["id"] == selected)
    a, b = st.columns(2)
    a.metric("Defect", str(detection["class_name"]).replace("_", " ").title())
    b.metric("Confidence", f"{float(detection['confidence']):.1%}")
    if st.button("Generate Explanation", key="gradcam_button"):
        try:
            with st.spinner("Computing neural activations and gradients..."):
                st.session_state.gradcam = generate_detection_gradcam(model, record["image_rgb"], detection["xyxy"], int(detection["class_id"]))
                st.session_state.gradcam_id = selected
        except GradCAMError as error: st.error(f"GRAD-CAM UNAVAILABLE: {error}")
        except Exception as error: st.error(f"GRAD-CAM SYSTEM ERROR: {error}")
    result = st.session_state.get("gradcam")
    if result is None or st.session_state.get("gradcam_id") != selected: return
    label = f"{detection['id']} {str(detection['class_name']).replace('_', ' ').title()}"
    context_image, context_cam, context_box = crop_detection_context(record["image_rgb"], result.cam, detection["xyxy"])
    boxed = draw_detection_box(context_image, context_box, label)
    overlay = draw_detection_box(blend_cam(context_image, context_cam), context_box, label)
    c1, c2, c3 = st.columns(3)
    c1.image(boxed, caption="DEFECT CONTEXT // SELECTED AREA", use_container_width=True)
    c2.image(colorize_cam(context_cam), caption="WHY THE MODEL FOCUSED HERE", use_container_width=True)
    c3.image(overlay, caption="MODEL ATTENTION // OVERLAY", use_container_width=True)
    st.info("How to read this: red/yellow regions had the strongest positive influence on the selected defect prediction; blue regions had little influence. The green box is the defect reported by YOLO.")
    with st.expander("View full-board Grad-CAM context"):
        full_boxed = draw_detection_box(record["image_rgb"], detection["xyxy"], label)
        full_overlay = draw_detection_box(blend_cam(record["image_rgb"], result.cam), detection["xyxy"], label)
        f1, f2 = st.columns(2)
        f1.image(full_boxed, caption="FULL PCB // SELECTED DETECTION", use_container_width=True)
        f2.image(full_overlay, caption="FULL PCB // MODEL ATTENTION", use_container_width=True)
    st.markdown("#### Explanation statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAM maximum", f"{result.max_activation:.3f}")
    m2.metric("Mean CAM inside box", f"{result.mean_inside_box:.3f}")
    m3.metric("Mean CAM outside box", f"{result.mean_outside_box:.3f}")
    m4.metric("CAM localization score", f"{result.localization_score:.1%}")
    if result.mean_inside_box > result.mean_outside_box:
        st.success("Interpretation: model attention is stronger inside the reported defect area than in the surrounding PCB.")
    else:
        st.warning("Interpretation: attention is diffuse or stronger outside this small defect box. Treat this prediction as a review cue, not strong localized visual evidence.")
    x1, y1, x2, y2 = map(float, detection["xyxy"])
    st.caption(f"Box: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}). Localization score is the fraction of total CAM activation inside this box, not accuracy. Target raw-anchor class score: {result.target_score:.3f}; feature layer: {result.layer_name}.")


st.markdown("<h1 style='color:#fff;margin-bottom:0'>PCB Defect Detection Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8c947d;font-size:14px;text-transform:uppercase;letter-spacing:1px'>Synthetic Inspector Prototype // V1.0</p>", unsafe_allow_html=True)
st.markdown("---")
left, right = st.columns([1.5, 1])
with right:
    st.markdown("### 01 / Control Panel")
    uploaded = st.file_uploader("Upload PCB Image (.jpg, .png)", type=["jpg", "jpeg", "png"])
    st.markdown("---"); st.markdown("### 02 / Telemetry")
    st.metric("Inference Device", "GPU (CUDA)" if torch.cuda.is_available() else "CPU", "Online")
    st.metric("Model Loaded", "Colour.pt", "Ready" if MODEL_PATH.is_file() else "Missing")
with left:
    st.markdown("### 03 / Visual Inspector")
    if uploaded:
        preview = Image.open(uploaded).convert("RGB")
        st.image(preview, caption="TARGET PCB // WAITING FOR INITIALIZATION", use_container_width=True)
        start = st.button("Start Inspection")
    else:
        start = False
        st.markdown("<div style='height:300px;display:flex;align-items:center;justify-content:center;background:#1a1a1a;border:1px dashed #353535;color:#8c947d;font-family:Space Grotesk'>NO IMAGE SIGNAL DETECTED</div>", unsafe_allow_html=True)
if start and uploaded:
    try:
        with st.spinner("Analyzing Defect Signatures..."):
            st.session_state.inspection = inspect(load_model(str(MODEL_PATH)), np.asarray(preview))
            st.session_state.pop("gradcam", None); st.session_state.pop("gradcam_id", None)
    except Exception as error: st.error(f"SYSTEM ERROR DURING INSPECTION: {error}")

record = st.session_state.get("inspection")
if record:
    try:
        model = load_model(str(MODEL_PATH)); detections = record["detections"]
        st.markdown("---"); st.success("INSPECTION COMPLETE")
        if detections:
            st.warning(f"CRITICAL: {len(detections)} DEFECTS DETECTED.")
        else:
            st.info("STATUS OK: NO DEFECTS DETECTED. PCB PASSES INSPECTION.")
        a, b = st.columns([1.5, 1]); a.image(record["annotated"], caption="TARGET PCB // SCAN RESULTS", use_container_width=True)
        if detections:
            b.markdown("### 04 / Defect Log")
            b.table(pd.DataFrame({"ID":[d["id"] for d in detections], "Defect Type":[str(d["class_name"]).replace("_", " ").title() for d in detections], "Confidence":[f"{float(d['confidence']):.2%}" for d in detections]}))
            a, b = st.columns([1.5, 1]); a.markdown("### 05 / Topographical Heatmap"); a.caption("Defect-density display derived from boxes — not Explainable AI."); a.image(density_heatmap(record["image_rgb"], detections), caption="TARGET PCB // DEFECT DENSITY", use_container_width=True)
            b.markdown("### 06 / Experiment Metrics"); b.table(pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1 Score","Specificity","mAP@50"], "Value":["96.4%","93.1%","95.8%","94.4%","97.2%","96.1%"]}))
            display_xai(model, record)
    except Exception as error: st.error(f"SYSTEM ERROR WHILE DISPLAYING RESULTS: {error}")
