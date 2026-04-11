# PCB Defect Detection Engine ⚙️

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**🚀 Live Website:** [https://pcb-defect-detection-engine.streamlit.app/](https://pcb-defect-detection-engine.streamlit.app/)

## Overview
The **PCB Defect Detection Engine** is an advanced computer vision dashboard built to identify and highlight manufacturing defects on Printed Circuit Boards (PCBs) in real-time. Powered by a custom-trained **YOLOv8** model, this application provides accurate bounding box classification for anomalies such as missing holes, spurious copper, mouse bites, and open circuits.

The interface is custom-engineered with a dark, brutalist "Titan Precision" design style inspired by NVIDIA, focusing heavily on high-contrast telemetry, strict geometry, and actionable visual metrics.

## Features
- **Real-Time Detection:** Upload a high-resolution target image (`.jpg`, `.png`) to instantly scan for anomalies.
- **YOLOv8 Inference Engine:** Fast and precise detection using state-of-the-art vision models tracking multiple defect classes.
- **Dynamic Defect Log:** Full readout array of localized defect types categorized alongside their exact confidence percentages natively positioned next to the scan results.
- **Industrial Dashboard UI:** Sleek, pure-black `#000000` theme with highly visible `#76b900` green accents.

## Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fantestico/PCB-Defect-Detection-Engine.git
   cd PCB-Defect-Detection-Engine
   ```

2. **Install dependencies:**
   Ensure you have multiple dependencies installed including `ultralytics`, `streamlit`, and `opencv-python-headless`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```
   
## Model Integration
The backend utilizes the `Colour.pt` weights file dynamically packaged in standard PyTorch format, relying on `ultralytics` for rapid CPU/GPU fallback inference processing.

---
*Developed for optimal precision and industrial reliability.*
