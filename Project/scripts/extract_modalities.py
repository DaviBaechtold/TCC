#!/usr/bin/env python
"""Extract multimodal inputs (keypoints, depth, segmentation mask, video clip) from a video file.

Pipeline (minimal placeholder):
  1. Read video frames with OpenCV
  2. Run MediaPipe holistic for keypoints (pose+hands) -> (T,J,2)
  3. Depth: placeholder using simple luminance normalization as pseudo-depth unless a real model is integrated
  4. Segmentation: basic threshold on depth proxy to create person mask (stub)
  5. Save arrays into NPZ.

This script is intentionally lightweight; to integrate real monocular depth or segmentation models,
replace the placeholder functions with calls to proper networks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

try:
    import mediapipe as mp
except Exception:
    mp = None

POSE_JOINTS = 33
HAND_JOINTS = 21
TOTAL_JOINTS = POSE_JOINTS + HAND_JOINTS*2


def extract_keypoints(frame, holistic) -> np.ndarray:
    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)
    pts = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan,np.nan]]*POSE_JOINTS)
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan,np.nan]]*HAND_JOINTS)
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan,np.nan]]*HAND_JOINTS)
    return np.array(pts, dtype=np.float32)


def pseudo_depth(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= 255.0
    return gray  # (H,W) in [0,1]


def pseudo_mask(depth_map: np.ndarray) -> np.ndarray:
    # simple threshold
    m = (depth_map > depth_map.mean()).astype(np.uint8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--max-frames', type=int, default=0, help='Limit number of frames (0 = all)')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--resize', type=int, nargs=2, default=None, help='Resize frames to WxH before processing')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    if mp is None:
        raise ImportError('mediapipe not installed')

    mp_holistic = mp.solutions.holistic
    keypoints = []
    depths = []
    masks = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride != 0:
                idx += 1; continue
            if args.max_frames and len(keypoints) >= args.max_frames:
                break
            if args.resize:
                w,h = args.resize
                frame = cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)
            kpts = extract_keypoints(frame, holistic)
            d = pseudo_depth(frame)
            m = pseudo_mask(d)
            keypoints.append(kpts)
            depths.append(d)
            masks.append(m)
            idx += 1
    cap.release()

    keypoints = np.stack(keypoints, axis=0)            # (T,J,2)
    depth_arr = np.stack(depths, axis=0)               # (T,H,W)
    mask_arr = np.stack(masks, axis=0)                 # (T,H,W)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keypoints=keypoints, depth=depth_arr, mask=mask_arr)
    print(f"Saved modalities to {args.out}: keypoints{keypoints.shape}, depth{depth_arr.shape}, mask{mask_arr.shape}")


if __name__ == '__main__':
    main()
