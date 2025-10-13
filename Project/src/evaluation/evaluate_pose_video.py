"""Compare RTMPose predictions on paired RGB/IR videos."""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Allow numpy-based checkpoints
try:
    import numpy as _np
    if hasattr(torch.serialization, "add_safe_globals"):
        safe_items: List[object] = []
        for mod_name in ("_core", "core"):
            try:
                candidate = getattr(getattr(_np, mod_name), "multiarray")._reconstruct
                safe_items.append(candidate)
            except Exception:
                continue
        if safe_items:
            torch.serialization.add_safe_globals(safe_items)
except Exception:
    pass

_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load

try:
    from mmpose.apis import init_model, inference_topdown
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"mmpose is required: {exc}")

try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    init_detector = None
    inference_detector = None


def extract_kps(result):
    if not result:
        return None
    collected = []
    for sample in result:
        if hasattr(sample, "pred_instances"):
            inst = sample.pred_instances
            if hasattr(inst, "keypoints"):
                keypoints = inst.keypoints
                if hasattr(keypoints, "cpu"):
                    keypoints = keypoints.cpu().numpy()
                else:
                    keypoints = np.asarray(keypoints)
                if keypoints.ndim == 2:
                    keypoints = keypoints[None, ...]
                collected.append(keypoints)
    if not collected:
        return None
    return np.concatenate(collected, axis=0)


def normalized_keypoint_distance(kp1: np.ndarray, kp2: np.ndarray) -> float:
    if kp1 is None or kp2 is None:
        return float("inf")
    if kp1.ndim == 3:
        kp1 = kp1[0]
    if kp2.ndim == 3:
        kp2 = kp2[0]
    torso = np.linalg.norm(kp1[5, :2] - kp1[6, :2])
    if torso < 1e-6:
        torso = 1.0
    dists = np.linalg.norm(kp1[:, :2] - kp2[:, :2], axis=1)
    return float(np.mean(dists) / torso)


def draw_instances(frame: np.ndarray, keypoints: Optional[np.ndarray], bboxes: Optional[np.ndarray], seed: Tuple[int, int]) -> None:
    if keypoints is None:
        return
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    rng_base = hash(seed) & 0xFFFFFFFF
    for idx, kp in enumerate(keypoints):
        rng = np.random.default_rng(rng_base + idx)
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        if bboxes is not None and idx < len(bboxes):
            box = bboxes[idx]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        for x, y in kp[:, :2]:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RTMPose on paired videos")
    parser.add_argument("--cfg", required=True, help="Pose config")
    parser.add_argument("--ckpt", required=True, help="Pose checkpoint")
    parser.add_argument("--rgb-video", required=True, help="RGB video path")
    parser.add_argument("--ir-video", required=False, default="", help="IR video path (optional)")
    parser.add_argument("--det-cfg", default="", help="Detector config (optional)")
    parser.add_argument("--det-ckpt", default="", help="Detector checkpoint")
    parser.add_argument("--det-score-thr", type=float, default=0.4, help="Detector score threshold")
    parser.add_argument("--device", default="cuda:0", help="Device id")
    parser.add_argument("--out-dir", default="work_dirs/video_eval", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit number of frames processed")
    parser.add_argument("--display", action="store_true", help="Show live preview")
    args = parser.parse_args()

    if not Path(args.rgb_video).is_file():
        raise SystemExit(f"RGB video not found: {args.rgb_video}")
    if args.ir_video and not Path(args.ir_video).is_file():
        raise SystemExit(f"IR video not found: {args.ir_video}")

    pose_model = init_model(args.cfg, args.ckpt, device=args.device)
    det_model = None
    if args.det_cfg and args.det_ckpt:
        if init_detector is None or inference_detector is None:
            raise SystemExit("mmdet is required when --det-cfg/--det-ckpt are set")
        det_model = init_detector(args.det_cfg, args.det_ckpt, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    rgb_cap = cv2.VideoCapture(args.rgb_video)
    ir_cap = None
    if args.ir_video:
        ir_cap = cv2.VideoCapture(args.ir_video)

    fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(os.path.join(args.out_dir, "rgb_overlay.mp4"), fourcc, fps, (width, height))
    ir_writer = None
    if ir_cap is not None:
        ir_writer = cv2.VideoWriter(os.path.join(args.out_dir, "ir_overlay.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    distances: List[float] = []

    try:
        while True:
            if args.max_frames >= 0 and frame_idx >= args.max_frames:
                break
            ret_rgb, rgb_frame = rgb_cap.read()
            if not ret_rgb:
                break
            ret_ir = False
            ir_frame = None
            if ir_cap is not None:
                ret_ir, ir_frame = ir_cap.read()
                if not ret_ir:
                    break
            if rgb_frame is None:
                break

            if det_model is not None:
                det_result = inference_detector(det_model, rgb_frame)
                if hasattr(det_result, "pred_instances"):
                    inst = det_result.pred_instances
                    boxes = inst.bboxes.cpu().numpy()
                    scores = inst.scores.cpu().numpy()
                    boxes = boxes[scores >= args.det_score_thr]
                else:
                    det_array = det_result[0]
                    boxes = det_array[det_array[:, 4] >= args.det_score_thr, :4] if det_array.size else np.empty((0, 4), dtype=np.float32)
                bboxes = boxes.astype(np.float32)
                if bboxes.size == 0:
                    bboxes = None
            else:
                h, w = rgb_frame.shape[:2]
                bboxes = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)

            rgb_results = inference_topdown(pose_model, rgb_frame, bboxes=bboxes)
            rgb_kps = extract_kps(rgb_results)

            if ir_frame is not None:
                ir_results = inference_topdown(pose_model, ir_frame, bboxes=bboxes)
                ir_kps = extract_kps(ir_results)
            else:
                ir_results = None
                ir_kps = None

            if rgb_kps is not None and ir_kps is not None:
                num_instances = min(len(rgb_kps), len(ir_kps))
                if num_instances > 0:
                    dist_vals = [normalized_keypoint_distance(rgb_kps[i], ir_kps[i]) for i in range(num_instances)]
                    distances.extend(dist_vals)
                    frame_dist = float(np.mean(dist_vals))
                else:
                    frame_dist = float("inf")
            else:
                frame_dist = float("inf")

            draw_instances(rgb_frame, rgb_kps, bboxes, (frame_idx, 0))
            if ir_frame is not None:
                draw_instances(ir_frame, ir_kps, bboxes, (frame_idx, 1))

            if frame_dist != float("inf"):
                cv2.putText(rgb_frame, f"Mean dist: {frame_dist:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if ir_frame is not None:
                    cv2.putText(ir_frame, f"Mean dist: {frame_dist:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            rgb_writer.write(rgb_frame)
            if ir_writer is not None and ir_frame is not None:
                ir_writer.write(ir_frame)

            if args.display:
                cv2.imshow("RGB Overlay", rgb_frame)
                if ir_frame is not None:
                    cv2.imshow("IR Overlay", ir_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        rgb_cap.release()
        if ir_cap is not None:
            ir_cap.release()
        rgb_writer.release()
        if ir_writer is not None:
            ir_writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("VIDEO EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {frame_idx}")
    if distances:
        arr = np.asarray(distances)
        print(f"Mean distance: {arr.mean():.4f}")
        print(f"Median distance: {np.median(arr):.4f}")
        print(f"Std distance: {arr.std():.4f}")
        print(f"Min distance: {arr.min():.4f}")
        print(f"Max distance: {arr.max():.4f}")
    else:
        print("No overlapping keypoints were evaluated")
    print(f"Outputs saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
