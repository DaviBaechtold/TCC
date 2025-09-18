#!/usr/bin/env python
"""Capture 2D keypoints from webcam (MediaPipe Holistic hands+pose) and save to .npz
Output arrays:
  pose2d: (N, J, 2) where J = 33 (pose) + 21 (left hand) + 21 (right hand) = 75 by default
You can later subset joints or map to your lifter's expected topology.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None

POSE_JOINTS = 33
HAND_JOINTS = 21


def extract_points(result, w, h):
    pts = []
    # Pose
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan, np.nan]] * POSE_JOINTS)
    # Left hand
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan, np.nan]] * HAND_JOINTS)
    # Right hand
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])
    else:
        pts.extend([[np.nan, np.nan]] * HAND_JOINTS)
    return np.array(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True, help='Output .npz path')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--show', action='store_true', help='Display annotated frames')
    args = ap.parse_args()

    if mp is None:
        raise ImportError('mediapipe not installed')

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    samples = []
    t0 = time.time()
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            pts = extract_points(result, w, h)
            samples.append(pts)
            if args.show:
                cv2.putText(frame, f"Frame {i+1}/{args.frames}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                cv2.imshow('capture', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
    if args.show:
        cv2.destroyAllWindows()
    arr = np.stack(samples, axis=0)  # (N,J,2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pose2d=arr)
    print(f"Saved {arr.shape} to {args.out} | elapsed {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
