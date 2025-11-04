#!/usr/bin/env python3
"""Quick test to get full traceback."""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown

print("Loading model...")
model = init_model(
    "work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py",
    "work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth",
    device='cuda:0'
)

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
h, w = frame.shape[:2]
full_bbox = np.array([[0, 0, w, h]])

print(f"Running inference with bbox shape: {full_bbox.shape}")
print(f"Bbox: {full_bbox}")

try:
    results = inference_topdown(model, frame, bboxes=full_bbox)
    print("SUCCESS!")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
