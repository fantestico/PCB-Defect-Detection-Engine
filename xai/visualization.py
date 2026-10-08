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


def crop_detection_context(
    image_rgb: np.ndarray, cam: np.ndarray, box_xyxy: Sequence[float], min_size: int = 280
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Crop the real CAM and image around a defect for human-scale review.

    This is only a shared crop of the original image and the computed CAM; it
    does not mask, alter, or regenerate the neural explanation.
    """
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box_xyxy[:4])
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    context_width = max(float(min_size), box_width * 5.0)
    context_height = max(float(min_size), box_height * 5.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, int(round(center_x - context_width / 2)))
    top = max(0, int(round(center_y - context_height / 2)))
    right = min(width, int(round(center_x + context_width / 2)))
    bottom = min(height, int(round(center_y + context_height / 2)))
    local_box = (x1 - left, y1 - top, x2 - left, y2 - top)
    return image_rgb[top:bottom, left:right], cam[top:bottom, left:right], local_box


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )
