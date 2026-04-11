import streamlit as st
import cv2
import numpy as np
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:
    import os
    os.system("pip install ultralytics opencv-python-headless pillow numpy")
    from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="PCB Defect Detection Engine", page_icon="⚙️")

# Inject custom CSS for NVIDIA aesthetic according to DESIGN.md
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* NVIDIA Green for accents */
    .stButton>button {
        border: 2px solid #76b900 !important;
        background: linear-gradient(135deg, rgba(85,133,0,0.8) 0%, rgba(118,185,0,0.2) 100%) !important;
        color: #ffffff !important;
        border-radius: 2px !important;
        font-weight: 700 !important;
        padding: 11px 13px !important;
        text-transform: uppercase;
        font-family: 'Space Grotesk', sans-serif;
        box-shadow: 0px 0px 5px rgba(118, 185, 0, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        box-shadow: 0px 0px 10px rgba(118, 185, 0, 0.5);
        border-color: #94da32 !important;
        color: #ffffff !important;
    }
    
    .stTextInput>div>div>input {
        border-radius: 2px;
        border-bottom: 2px solid #353535;
        background-color: transparent !important;
        color: #ffffff !important;
    }
    .stTextInput>div>div>input:focus {
        border-bottom-color: #76b900;
        box-shadow: none !important;
    }
    
    /* Strict industrial containers */
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        padding: 1rem;
        border-left: 2px solid #76b900;
        border-radius: 2px;
        box-shadow: rgba(0, 0, 0, 0.3) 0px 0px 5px 0px;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    /* Table styling for dark theme */
    thead tr th {
        background-color: #1a1a1a !important;
        color: #76b900 !important;
        font-family: 'Space Grotesk', sans-serif;
        border-bottom: 1px solid #353535 !important;
        border-right: none !important;
        border-left: none !important;
        border-top: none !important;
    }
    tbody tr td {
        background-color: transparent !important;
        border-bottom: 1px solid #353535 !important;
        border-right: none !important;
        border-left: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<h1 style='color: #ffffff; margin-bottom: 0;'>PCB Defect Detection Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8c947d; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;'>Synthetic Inspector Prototype // V1.0</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout
col_img, col_data = st.columns([1.5, 1])

with col_data:
    st.markdown("### 01 / Control Panel")
    uploaded_file = st.file_uploader("Upload PCB Image (.jpg, .png)", type=['jpg', 'jpeg', 'png'])
    
    st.markdown("---")
    st.markdown("### 02 / Telemetry")
    try:
        import torch
        device_status = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except:
        device_status = "CPU"
        
    st.metric(label="Inference Device", value=device_status, delta="Online", delta_color="normal")
    st.metric(label="Model Loaded", value="Colour.pt", delta="Ready", delta_color="normal")
    
    table_placeholder = st.empty()

with col_img:
    st.markdown("### 03 / Visual Inspector")
    do_inspect = False
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='TARGET PCB // WAITING FOR INITIALIZATION', use_container_width=True)
        do_inspect = st.button("Start Inspection")
    else:
        # Placeholder box
        st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; background-color: #1a1a1a; border: 1px dashed #353535; border-radius: 2px; color: #8c947d; font-family: Space Grotesk;'>NO IMAGE SIGNAL DETECTED</div>", unsafe_allow_html=True)

# Post-Inspection Results Area (Spanning Full Width to maintain size and align horizontally)
if do_inspect:
    st.markdown("---")
    with st.spinner("Analyzing Defect Signatures..."):
        try:
            import os
            if not os.path.exists("Colour.pt"):
                st.error("Model file 'Colour.pt' not found in the current directory.")
            else:
                model = YOLO("Colour.pt")
                results = model.predict(image)
                res_plotted = results[0].plot()
                
                boxes = results[0].boxes
                
                st.success("INSPECTION COMPLETE")
                if len(boxes) > 0:
                    st.warning(f"CRITICAL: {len(boxes)} DEFECTS DETECTED.")
                else:
                    st.info("STATUS OK: NO DEFECTS DETECTED. PCB PASSES INSPECTION.")
                
                # Re-create the 1.5 : 1 ratio so the result image matches the preview image's size
                res_col_img, res_col_data = st.columns([1.5, 1])
                
                with res_col_img:
                    st.image(res_plotted, caption='TARGET PCB // SCAN RESULTS', use_container_width=True)
                    
                with res_col_data:
                    if len(boxes) > 0:
                        # Create a clean data table
                        import pandas as pd
                        data = {
                            "ID": [f"DEF-{i:03d}" for i in range(1, len(boxes) + 1)],
                            "Defect Type": [model.names[int(c)] for c in boxes.cls], 
                            "Confidence": [f"{conf:.2%}" for conf in boxes.conf]
                        }
                        df = pd.DataFrame(data)
                        st.markdown("### 04 / Defect Log")
                        st.table(df)
        except Exception as e:
            st.error(f"SYSTEM ERROR DURING INSPECTION: {e}")
