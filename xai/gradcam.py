"""Detection-specific Grad-CAM for compatible Ultralytics YOLO detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GradCAMError(RuntimeError):
    """Raised when a checkpoint cannot provide a reliable Grad-CAM target."""


@dataclass(frozen=True)
class GradCAMResult:
    """The normalized CAM and transparent explanation statistics."""

    cam: np.ndarray
    target_anchor: int
    target_score: float
    layer_name: str
    max_activation: float
    mean_inside_box: float
    mean_outside_box: float
    localization_score: float


def generate_detection_gradcam(
    yolo_model: Any,
    image_rgb: np.ndarray,
    box_xyxy: Sequence[float],
    class_id: int,
) -> GradCAMResult:
    """Generate Grad-CAM for one post-NMS YOLO detection.

    The backward objective is the predicted class score for the raw anchor whose
    decoded box best matches the selected post-NMS detection.  This keeps the
    explanation object-specific instead of using a global class objective.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise GradCAMError("Grad-CAM requires an RGB image with three channels.")

    core = getattr(yolo_model, "model", yolo_model)
    if not isinstance(core, nn.Module):
        raise GradCAMError("The loaded model does not expose a PyTorch detection module.")

    device = next(core.parameters(), None)
    if device is None:
        raise GradCAMError("The detection model has no parameters to differentiate.")
    device = device.device
    core.eval()
    # Ultralytics prediction may freeze parameters for inference. Grad-CAM needs
    # an autograd graph through the feature layer, but does not update weights.
    parameter_states = [parameter.requires_grad for parameter in core.parameters()]
    for parameter in core.parameters():
        parameter.requires_grad_(True)

    target_layer, layer_name = _find_target_layer(core)
    input_tensor, ratio, pad_x, pad_y = _preprocess(image_rgb, _model_input_size(core))
    input_tensor = input_tensor.to(device)
    projected_box = _project_box_to_model(box_xyxy, ratio, pad_x, pad_y)

    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor):
            raise GradCAMError("The selected convolutional layer did not return a tensor.")
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        core.zero_grad(set_to_none=True)
        raw = _prediction_tensor(core(input_tensor))
        boxes, class_scores = _decode_prediction(raw, int(class_id))
        anchor_index = _select_anchor(boxes, class_scores, projected_box)
        target = class_scores[anchor_index]
        if not torch.isfinite(target) or not target.requires_grad:
            raise GradCAMError("The selected detection score is not differentiable.")
        target.backward()
    except GradCAMError:
        raise
    except Exception as error:
        raise GradCAMError(f"Could not calculate gradients for this detection: {error}") from error
    finally:
        handle.remove()
        for parameter, required in zip(core.parameters(), parameter_states):
            parameter.requires_grad_(required)

    if not activations or not gradients:
        raise GradCAMError("No activations or gradients were captured from the selected layer.")
    activation, gradient = activations[-1], gradients[-1]
    if activation.ndim != 4 or gradient.ndim != 4:
        raise GradCAMError("The selected layer does not provide spatial convolutional features.")

    # Grad-CAM: global-average-pool gradients, then weight feature maps by them.
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam_tensor = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam_tensor = F.interpolate(
        cam_tensor, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam_tensor.detach().float().cpu().numpy()
    cam = _normalize(cam)
    stats = _cam_statistics(cam, box_xyxy)
    return GradCAMResult(
        cam=cam,
        target_anchor=int(anchor_index),
        target_score=float(target.detach().cpu()),
        layer_name=layer_name,
        **stats,
    )


def _find_target_layer(model: nn.Module) -> tuple[nn.Conv2d, str]:
    """Choose a high-resolution feature convolution used by the Detect head."""
    layers = getattr(model, "model", None)
    head = layers[-1] if isinstance(layers, nn.Sequential) and len(layers) else None
    feature_indices = getattr(head, "f", None)
    if isinstance(feature_indices, (list, tuple)) and layers is not None:
        # The first detector feature has the finest spatial grid in YOLO heads.
        feature_layer = layers[int(feature_indices[0])]
        convs = [module for module in feature_layer.modules() if isinstance(module, nn.Conv2d)]
        if convs:
            return convs[-1], f"model.{feature_indices[0]}.{convs[-1].__class__.__name__}"

    # Fallback: latest convolution outside the terminal prediction head.
    candidates: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not name.startswith("model.-1"):
            candidates.append((name, module))
    if not candidates:
        raise GradCAMError("No suitable convolutional feature layer was found for Grad-CAM.")
    return candidates[-1]


def _model_input_size(model: nn.Module) -> tuple[int, int]:
    args = getattr(model, "args", {}) or {}
    size = args.get("imgsz", 640) if isinstance(args, dict) else 640
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 640, 640


def _preprocess(image_rgb: np.ndarray, size: tuple[int, int]) -> tuple[Tensor, float, int, int]:
    input_h, input_w = size
    original_h, original_w = image_rgb.shape[:2]
    ratio = min(input_w / original_w, input_h / original_h)
    resized_w, resized_h = round(original_w * ratio), round(original_h * ratio)
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (input_w - resized_w) // 2, (input_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return tensor, ratio, pad_x, pad_y


def _prediction_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise GradCAMError("Unsupported YOLO output structure; no prediction tensor was returned.")


def _decode_prediction(raw: Tensor, class_id: int) -> tuple[Tensor, Tensor]:
    """Read eval-mode YOLOv8 decoded boxes and one class-score row."""
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise GradCAMError(f"Unsupported prediction shape {tuple(raw.shape)}.")
    prediction = raw[0]
    # Ultralytics YOLOv8 produces [4 + num_classes, num_anchors] in eval mode.
    if prediction.shape[0] >= 5 and prediction.shape[0] <= prediction.shape[1]:
        if class_id < 0 or 4 + class_id >= prediction.shape[0]:
            raise GradCAMError("The selected class is not present in the model output.")
        boxes_xywh, scores = prediction[:4].transpose(0, 1), prediction[4 + class_id]
    else:
        # Support equivalent [num_anchors, 4 + num_classes] output layouts.
        if class_id < 0 or 4 + class_id >= prediction.shape[1]:
            raise GradCAMError("The selected class is not present in the model output.")
        boxes_xywh, scores = prediction[:, :4], prediction[:, 4 + class_id]
    return _xywh_to_xyxy(boxes_xywh), scores


def _select_anchor(boxes: Tensor, class_scores: Tensor, target_box: Tensor) -> int:
    iou = _box_iou(boxes, target_box.unsqueeze(0)).squeeze(1)
    matching_score = iou * class_scores.sigmoid() if class_scores.max() > 1 else iou * class_scores
    index = int(matching_score.argmax().detach().cpu())
    if float(iou[index].detach().cpu()) < 0.05:
        raise GradCAMError("Could not match the selected detection to a raw YOLO prediction.")
    return index


def _project_box_to_model(box: Sequence[float], ratio: float, pad_x: int, pad_y: int) -> Tensor:
    values = torch.tensor(box[:4], dtype=torch.float32)
    values[0::2] = values[0::2] * ratio + pad_x
    values[1::2] = values[1::2] * ratio + pad_y
    return values


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(dim=1)
    return torch.stack((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), dim=1)


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    intersection_lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    intersection_rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (intersection_rb - intersection_lt).clamp(min=0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-7)


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = cam - float(cam.min())
    maximum = float(cam.max())
    return cam / maximum if maximum > 1e-12 else np.zeros_like(cam)


def _cam_statistics(cam: np.ndarray, box: Sequence[float]) -> dict[str, float]:
    height, width = cam.shape
    x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    mask = np.zeros_like(cam, dtype=bool)
    mask[y1:y2, x1:x2] = True
    inside = cam[mask]
    outside = cam[~mask]
    total = float(cam.sum())
    return {
        "max_activation": float(cam.max()),
        "mean_inside_box": float(inside.mean()) if inside.size else 0.0,
        "mean_outside_box": float(outside.mean()) if outside.size else 0.0,
        "localization_score": float(inside.sum() / total) if total > 1e-12 else 0.0,
    }
