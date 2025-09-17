#!/usr/bin/env python
"""
Capture a single frame from the default webcam, run MediaPipe Hands, draw landmarks,
and save an example image for slides.

Usage:
  python Project/scripts/save_mediapipe_example.py --out Doc/Slides/mediapipe_example.jpg --hands 1 --mirror
"""
import argparse
from pathlib import Path
import sys

import cv2

try:
    import mediapipe as mp  # type: ignore
except Exception as e:
    mp = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='Doc/Slides/mediapipe_example.jpg')
    ap.add_argument('--hands', type=int, default=1)
    ap.add_argument('--mirror', action='store_true', help='Flip horizontally for selfie view')
    args = ap.parse_args()

    if mp is None:
        print('ERROR: mediapipe is not installed. pip install mediapipe')
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: could not open webcam')
        sys.exit(2)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=args.hands,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ok, frame = cap.read()
        if not ok:
            print('ERROR: could not read frame from webcam')
            sys.exit(3)
        if args.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS, draw_spec, draw_spec)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        print(f'Saved MediaPipe example to {out_path}')

    cap.release()


if __name__ == '__main__':
    main()
