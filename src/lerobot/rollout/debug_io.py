#!/usr/bin/env python

"""Optional rollout policy I/O logging for deployment debugging."""

from __future__ import annotations

import json
import os
import re
import time
from threading import Lock
from typing import Any

_DEBUG_IO_STEP = 0
_DEBUG_IO_LOCK = Lock()


def _is_camera_key(key_path: str) -> bool:
    key_path = key_path.lower()
    return "camera" in key_path or "image" in key_path


def _as_numpy(value: Any):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(value, "shape"):
        return value
    return None


def _first_image(array: Any):
    """Return one HWC/CHW image from common image, batch-image, or time-batch-image shapes."""
    if not hasattr(array, "ndim") or array.ndim < 3:
        return None
    img = array
    while img.ndim > 3:
        img = img[0]
    return img


def _image_layout(img: Any) -> str:
    if img.ndim != 3:
        return "unknown"
    if img.shape[0] in (1, 3, 4):
        return "CHW"
    if img.shape[-1] in (1, 3, 4):
        return "HWC"
    return "unknown"


def _channel_stats(img: Any, layout: str) -> dict[str, Any] | None:
    try:
        import numpy as np

        if layout == "CHW":
            channels = img
        elif layout == "HWC":
            channels = np.moveaxis(img, -1, 0)
        else:
            return None
        return {
            "channel_mean": [float(np.mean(c)) for c in channels],
            "channel_min": [float(np.min(c)) for c in channels],
            "channel_max": [float(np.max(c)) for c in channels],
        }
    except Exception:
        return None


def _save_camera_image(img: Any, key_path: str) -> str | None:
    directory = os.environ.get("LEROBOT_ROLLOUT_DEBUG_CAMERA_DIR")
    if not directory:
        return None
    try:
        import numpy as np
        from PIL import Image

        layout = _image_layout(img)
        if layout == "CHW":
            img = np.moveaxis(img, 0, -1)
        elif layout != "HWC":
            return None

        img = np.asarray(img)
        if img.shape[-1] == 1:
            img = img[..., 0]
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

        os.makedirs(directory, exist_ok=True)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_path).strip("_")
        path = os.path.join(directory, f"{_DEBUG_IO_STEP:06d}_{safe_key}.png")
        Image.fromarray(img).save(path)
        return path
    except Exception as exc:
        return f"save_failed: {exc}"


def _camera_debug_value(value: Any, key_path: str) -> Any | None:
    array = _as_numpy(value)
    if array is None:
        return None
    img = _first_image(array)
    if img is None:
        return None
    try:
        import numpy as np

        layout = _image_layout(img)
        summary = {
            "kind": "camera",
            "shape": list(array.shape),
            "first_image_shape": list(img.shape),
            "layout": layout,
            "dtype": str(array.dtype),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }
        stats = _channel_stats(img, layout)
        if stats:
            summary.update(stats)
        saved = _save_camera_image(img, key_path)
        if saved:
            summary["saved_image"] = saved
        return summary
    except Exception:
        return None


def _debug_value(value: Any, key_path: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _debug_value(v, f"{key_path}.{k}" if key_path else str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(v, f"{key_path}[{i}]") for i, v in enumerate(value)]

    if _is_camera_key(key_path):
        camera_summary = _camera_debug_value(value, key_path)
        if camera_summary is not None:
            return camera_summary

    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() <= 128:
                return value.tolist()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.float().mean()),
            }
    except Exception:
        pass

    if hasattr(value, "shape"):
        if value.size <= 128:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
        }
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def action_by_name(action: Any, names: list[str]) -> dict[str, Any] | None:
    try:
        values = action.detach().cpu().tolist()
    except AttributeError:
        try:
            values = action.tolist()
        except AttributeError:
            values = action
    if not isinstance(values, list):
        return None
    if values and isinstance(values[0], list):
        values = values[0]
    if len(values) != len(names):
        return None
    return {name: values[i] for i, name in enumerate(names)}


def maybe_debug_policy_io(payload: dict[str, Any]) -> None:
    global _DEBUG_IO_STEP
    path = os.environ.get("LEROBOT_ROLLOUT_DEBUG_IO")
    if not path:
        return
    every = max(1, int(os.environ.get("LEROBOT_ROLLOUT_DEBUG_EVERY", "1")))
    with _DEBUG_IO_LOCK:
        _DEBUG_IO_STEP += 1
        if _DEBUG_IO_STEP % every != 0:
            return

        record = {"step": _DEBUG_IO_STEP, "time": time.time()}
        record.update({k: _debug_value(v, str(k)) for k, v in payload.items()})
        line = json.dumps(record)
        if path in {"-", "stdout", "terminal"}:
            print(f"LEROBOT_DEBUG_IO {line}", flush=True)
            return
        with open(path, "a") as f:
            f.write(line + "\n")
