"""Real-time RTMPose inference on webcam or video source."""

import argparse
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# Allow loading checkpoints that rely on numpy internals.
# Monkey-patch torch.load to use weights_only=False by default (for compatibility with old checkpoints)
_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown  # type: ignore

try:
    from mmdet.apis import inference_detector, init_detector  # type: ignore
except Exception:  # pragma: no cover - detection is optional
    inference_detector = None
    init_detector = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RTMPose in real time.")
    parser.add_argument("--cfg", type=str, required=True, help="Pose config path.")
    parser.add_argument("--ckpt", type=str, required=True, help="Pose checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string (e.g. cuda:0 or cpu).")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path.")
    parser.add_argument("--width", type=int, default=0, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=0, help="Optional capture height.")
    parser.add_argument("--det-cfg", type=str, default="", help="Optional detector config (RTMDet or similar).")
    parser.add_argument("--det-ckpt", type=str, default="", help="Optional detector checkpoint.")
    parser.add_argument("--score-thr", type=float, default=0.4, help="Keypoint score threshold for drawing.")
    parser.add_argument("--bbox-thr", type=float, default=0.5, help="Detector score threshold.")
    return parser.parse_args()


def list_available_cameras(max_test: int = 10) -> list:
    """List available camera indices."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def open_source(source: str) -> cv2.VideoCapture:
    if len(source) == 1 and source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def get_full_frame_bbox(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)


def detect_person_bboxes(detector, frame: np.ndarray, score_thr: float) -> np.ndarray:
    result = inference_detector(detector, frame)
    if isinstance(result, tuple):
        bboxes = result[0]
    else:
        bboxes = result
    # Assume person class is index 0
    if isinstance(bboxes, list):
        if not bboxes:
            return np.empty((0, 4), dtype=np.float32)
        person_bboxes = bboxes[0]
    else:
        person_bboxes = bboxes
    if person_bboxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    person_bboxes = np.asarray(person_bboxes)
    if person_bboxes.shape[1] == 5:
        mask = person_bboxes[:, 4] >= score_thr
        person_bboxes = person_bboxes[mask, :4]
    return person_bboxes.astype(np.float32)


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray, scores: Optional[np.ndarray], meta: dict, score_thr: float) -> None:
    color = (0, 255, 0)
    skeleton: Sequence
    skeleton = meta.get("skeleton", [])
    if "skeleton_links" in meta and meta["skeleton_links"]:
        # Handle both dict format (link["link"]) and tuple/list format
        links = meta["skeleton_links"]
        if links and isinstance(links[0], dict):
            skeleton = [link["link"] for link in links]
        else:
            skeleton = links
    point_radius = 3
    thickness = 2
    pts = keypoints[:, :2]
    kp_scores = scores if scores is not None else np.ones(len(pts), dtype=np.float32)
    for idx, (x, y) in enumerate(pts):
        if kp_scores[idx] < score_thr:
            continue
        cv2.circle(frame, (int(x), int(y)), point_radius, color, -1)
    for link in skeleton:
        if isinstance(link, (list, tuple)) and len(link) >= 2:
            i, j = link[0], link[1]
        else:
            continue
        if i >= len(kp_scores) or j >= len(kp_scores):
            continue
        if kp_scores[i] < score_thr or kp_scores[j] < score_thr:
            continue
        pt1 = (int(pts[i, 0]), int(pts[i, 1]))
        pt2 = (int(pts[j, 0]), int(pts[j, 1]))
        cv2.line(frame, pt1, pt2, color, thickness)


def main() -> None:
    args = parse_args()

    pose_model = init_model(args.cfg, args.ckpt, device=args.device)
    det_model = None
    if args.det_cfg and args.det_ckpt:
        if init_detector is None or inference_detector is None:
            raise RuntimeError("mmdet is not available; install mmdet to use detection.")
        det_model = init_detector(args.det_cfg, args.det_ckpt, device=args.device)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"❌ Could not open source {args.source}")
        if args.source.isdigit():
            print("\n🔍 Scanning for available cameras...")
            available = list_available_cameras()
            if available:
                print(f"✅ Found cameras at indices: {available}")
                print(f"\nTry running with: --source {available[0]}")
            else:
                print("❌ No cameras found!")
                print("\nPossible solutions:")
                print("  1. Check if camera is connected")
                print("  2. Check if another program is using the camera")
                print("  3. Try running: ls /dev/video*")
                print("  4. Use a video file instead: --source path/to/video.mp4")
        raise RuntimeError(f"Could not open source {args.source}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    meta = getattr(pose_model, "dataset_meta", {})

    prev_time = time.time()
    fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = ensure_bgr(frame)

            if det_model is not None:
                bboxes = detect_person_bboxes(det_model, frame, args.bbox_thr)
                if bboxes.size == 0:
                    cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("RTMPose Real-Time", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
            else:
                bboxes = get_full_frame_bbox(frame)

            results = inference_topdown(pose_model, frame, bboxes=bboxes)
            for result in results:
                inst = result.pred_instances
                kps = inst.keypoints
                scores = getattr(inst, "keypoint_scores", None)
                if hasattr(kps, "cpu"):
                    kps_np = kps.cpu().numpy()
                else:
                    kps_np = np.asarray(kps)
                if scores is not None and hasattr(scores, "cpu"):
                    scores_np = scores.cpu().numpy()
                elif scores is None:
                    scores_np = None
                else:
                    scores_np = np.asarray(scores)
                for i in range(kps_np.shape[0]):
                    draw_keypoints(frame, kps_np[i], None if scores_np is None else scores_np[i], meta, args.score_thr)

            now = time.time()
            if now != prev_time:
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("RTMPose Real-Time", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
