"""Rendering helpers for scientifically generated CAM arrays."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def draw_detection_box(
    image_rgb: np.ndarray, box_xyxy: Sequence[float], label: str
) -> np.ndarray:
    """Return an RGB copy with the selected detection kept visible."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    x1, y1, x2, y2 = _clip_box(box_xyxy, width, height)
    cv2.rectangle(output, (x1, y1), (x2, y2), (118, 185, 0), 2)
    cv2.putText(
        output,
        label,
        (x1, max(18, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (118, 185, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def colorize_cam(cam: np.ndarray) -> np.ndarray:
    """Map normalized CAM values to a cool-to-warm JET heatmap in RGB."""
    cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def blend_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    """Blend a real model CAM over its source image."""
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colorize_cam(cam), alpha, 0)


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )
