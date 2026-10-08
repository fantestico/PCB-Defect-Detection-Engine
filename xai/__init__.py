"""Model-based explanation methods for the PCB inspection dashboard."""

from .gradcam import GradCAMError, GradCAMResult, generate_detection_gradcam

__all__ = ["GradCAMError", "GradCAMResult", "generate_detection_gradcam"]
