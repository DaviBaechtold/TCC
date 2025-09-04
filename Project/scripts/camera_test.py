#!/usr/bin/env python
"""
Webcam test: draw 2D landmarks and bounding boxes for Face, Hands, and Body.

Keys:
  q - quit
  f/h/p - toggle Face/Hands/Pose overlays
"""
import argparse
from typing import Iterable, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    mp = None


def _to_px(landmarks, w: int, h: int) -> np.ndarray:
    """Convert normalized landmarks to pixel coordinates [N,2]."""
    pts = []
    for lm in landmarks:
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)


def _bbox_from_points(pts: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    if pts.size == 0:
        return 0, 0, 0, 0
    x1, y1 = np.nanmin(pts, axis=0)
    x2, y2 = np.nanmax(pts, axis=0)
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--mirror', action='store_true', help='Flip horizontally for a selfie view')
    args = ap.parse_args()

    if mp is None:
        raise ImportError('mediapipe not installed. pip install mediapipe')

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    show_face, show_hands, show_pose = True, True, True
    prev_t = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            # Draw landmarks using built-in helpers
            if show_face and result.face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                pts = _to_px(result.face_landmarks.landmark, w, h)
                x1, y1, x2, y2 = _bbox_from_points(pts, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if show_hands:
                for hand_lms in (result.left_hand_landmarks, result.right_hand_landmarks):
                    if hand_lms is None:
                        continue
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    pts = _to_px(hand_lms.landmark, w, h)
                    x1, y1, x2, y2 = _bbox_from_points(pts, w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

            if show_pose and result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
                pts = _to_px(result.pose_landmarks.landmark, w, h)
                x1, y1, x2, y2 = _bbox_from_points(pts, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # FPS
            cur_t = cv2.getTickCount()
            dt = (cur_t - prev_t) / tick_freq
            prev_t = cur_t
            fps = 1.0 / max(1e-6, dt)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"[f]ace:{int(show_face)} [h]ands:{int(show_hands)} [p]ose:{int(show_pose)}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow('Camera Test (Holistic)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_face = not show_face
            elif key == ord('h'):
                show_hands = not show_hands
            elif key == ord('p'):
                show_pose = not show_pose

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
