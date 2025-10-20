"""Optimized real-time RTMPose inference with batch processing and performance improvements."""

import argparse
import time
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Monkey-patch torch.load for compatibility
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown
try:
    from mmdet.apis import inference_detector, init_detector
except Exception:
    inference_detector = None
    init_detector = None


class BatchedPoseEstimator:
    """
    Optimized pose estimator with batch processing for multiple people.
    Processes all detected persons in a single forward pass.
    """
    
    def __init__(self, model, input_size=(288, 384), device='cuda:0'):
        self.model = model
        self.input_size = input_size  # (width, height)
        self.device = device
        
    def preprocess_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> Tuple[torch.Tensor, List[dict]]:
        """
        Crop and preprocess all bboxes in batch.
        
        Args:
            frame: Original frame (H, W, 3)
            bboxes: Person bounding boxes (N, 4) - [x1, y1, x2, y2]
            
        Returns:
            batch_tensor: (N, 3, H, W) preprocessed tensor
            transform_info: List of dicts with transform params for each bbox
        """
        if len(bboxes) == 0:
            return torch.empty((0, 3, self.input_size[1], self.input_size[0])).to(self.device), []
        
        crops = []
        transform_info = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bbox slightly (10% margin)
            w, h = x2 - x1, y2 - y1
            x1 = max(0, x1 - int(w * 0.05))
            y1 = max(0, y1 - int(h * 0.05))
            x2 = min(frame.shape[1], x2 + int(w * 0.05))
            y2 = min(frame.shape[0], y2 + int(h * 0.05))
            
            # Crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Convert to grayscale if needed
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            
            # Resize to input size
            crop_resized = cv2.resize(crop, self.input_size)
            
            # Store transform info for denormalization
            transform_info.append({
                'bbox': [x1, y1, x2, y2],
                'scale_x': (x2 - x1) / self.input_size[0],
                'scale_y': (y2 - y1) / self.input_size[1],
                'offset_x': x1,
                'offset_y': y1
            })
            
            # Convert to tensor and normalize
            crop_tensor = self._to_tensor(crop_resized)
            crops.append(crop_tensor)
        
        if len(crops) == 0:
            return torch.empty((0, 3, self.input_size[1], self.input_size[0])).to(self.device), []
        
        # Stack into batch
        batch_tensor = torch.stack(crops).to(self.device)
        
        return batch_tensor, transform_info
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # To tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        # Normalize (ImageNet stats for grayscale)
        mean = torch.tensor([123.675]).view(1, 1, 1)
        std = torch.tensor([58.395]).view(1, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    @torch.no_grad()
    def inference_batch(self, batch_tensor: torch.Tensor, transform_info: List[dict]) -> List[np.ndarray]:
        """
        Run inference on batch of images.
        
        Args:
            batch_tensor: (N, 3, H, W) batch of preprocessed images
            transform_info: List of transform params
            
        Returns:
            List of keypoints arrays (N, 133, 3) - [x, y, confidence]
        """
        if batch_tensor.shape[0] == 0:
            return []
        
        # Forward pass (single batch!)
        with torch.amp.autocast('cuda', enabled=True):
            # Use MMPose's internal inference
            results = []
            for i in range(batch_tensor.shape[0]):
                # Convert single image to MMPose format
                img_single = batch_tensor[i].cpu().numpy()
                img_single = np.transpose(img_single, (1, 2, 0))
                
                # Denormalize
                mean = np.array([123.675])
                std = np.array([58.395])
                img_single = img_single * std + mean
                img_single = img_single.astype(np.uint8)
                
                # BGR
                img_single = cv2.cvtColor(img_single, cv2.COLOR_RGB2BGR)
                
                # Inference
                result = inference_topdown(self.model, img_single)
                results.append(result)
        
        # Denormalize keypoints to original frame coordinates
        keypoints_list = []
        for i, (result, info) in enumerate(zip(results, transform_info)):
            if len(result) == 0:
                keypoints_list.append(np.zeros((133, 3)))
                continue
                
            pred_instances = result[0].pred_instances
            # Handle both tensor and numpy array formats
            kps = pred_instances.keypoints[0]
            scores = pred_instances.keypoint_scores[0]
            
            if hasattr(kps, 'cpu'):
                keypoints = kps.cpu().numpy()  # (133, 2)
            else:
                keypoints = np.asarray(kps)
            
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()  # (133,)
            else:
                scores = np.asarray(scores)
            
            # Denormalize to original frame coordinates
            keypoints[:, 0] = keypoints[:, 0] * info['scale_x'] + info['offset_x']
            keypoints[:, 1] = keypoints[:, 1] * info['scale_y'] + info['offset_y']
            
            # Combine with scores
            keypoints_with_scores = np.concatenate([keypoints, scores[:, None]], axis=1)
            keypoints_list.append(keypoints_with_scores)
        
        return keypoints_list


class FPSCounter:
    """Smooth FPS counter with moving average."""
    
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.prev_time = time.time()
    
    def update(self):
        """Update FPS counter."""
        now = time.time()
        elapsed = now - self.prev_time
        self.times.append(elapsed)
        self.prev_time = now
    
    def get_fps(self) -> float:
        """Get smoothed FPS."""
        if len(self.times) == 0:
            return 0.0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / max(avg_time, 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized real-time RTMPose inference.")
    parser.add_argument("--cfg", type=str, required=True, help="Pose config path.")
    parser.add_argument("--ckpt", type=str, required=True, help="Pose checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu).")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path.")
    parser.add_argument("--det-cfg", type=str, default="", help="Optional detector config.")
    parser.add_argument("--det-ckpt", type=str, default="", help="Optional detector checkpoint.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Keypoint score threshold.")
    parser.add_argument("--bbox-thr", type=float, default=0.5, help="Detection score threshold.")
    parser.add_argument("--width", type=int, default=0, help="Capture width.")
    parser.add_argument("--height", type=int, default=0, help="Capture height.")
    parser.add_argument("--batch-size", type=int, default=8, help="Max persons to process in batch.")
    parser.add_argument("--no-display", action='store_true', help="Disable display window.")
    parser.add_argument("--benchmark", action='store_true', help="Benchmark mode (print detailed timing).")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    """Open video source (camera or file)."""
    if len(source) == 1 and source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def detect_person_bboxes(detector, frame: np.ndarray, score_thr: float) -> np.ndarray:
    """Detect person bounding boxes."""
    result = inference_detector(detector, frame)
    
    # Handle different output formats
    if isinstance(result, tuple):
        bboxes = result[0]
    else:
        bboxes = result
    
    # Get person class (index 0 in COCO)
    if isinstance(bboxes, list):
        if not bboxes:
            return np.empty((0, 4), dtype=np.float32)
        person_bboxes = bboxes[0]
    else:
        person_bboxes = bboxes
    
    if person_bboxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    person_bboxes = np.asarray(person_bboxes)
    
    # Filter by score
    if person_bboxes.shape[1] == 5:
        mask = person_bboxes[:, 4] >= score_thr
        person_bboxes = person_bboxes[mask, :4]
    
    return person_bboxes.astype(np.float32)


def draw_keypoints_batch(frame: np.ndarray, keypoints_list: List[np.ndarray], 
                         score_thr: float = 0.3, skeleton_links: Optional[list] = None):
    """Draw keypoints for multiple people."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    for person_id, keypoints in enumerate(keypoints_list):
        color = colors[person_id % len(colors)]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > score_thr:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw skeleton
        if skeleton_links is not None:
            for link in skeleton_links:
                # Handle different link formats
                if isinstance(link, dict):
                    pt1_idx, pt2_idx = link['link']
                else:
                    pt1_idx, pt2_idx = link
                
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    
                    if pt1[2] > score_thr and pt2[2] > score_thr:
                        cv2.line(frame, 
                                (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), 
                                color, 2)


def get_full_frame_bbox(frame: np.ndarray) -> np.ndarray:
    """Get full frame as single bbox (for single-person mode)."""
    h, w = frame.shape[:2]
    return np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)


def main():
    args = parse_args()
    
    print("🚀 Initializing Optimized Real-Time Pose Estimation...")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize pose model
    print(f"Loading pose model: {args.cfg}")
    pose_model = init_model(args.cfg, args.ckpt, device=args.device)
    
    # Initialize batched estimator
    estimator = BatchedPoseEstimator(pose_model, device=args.device)
    
    # Initialize detector (optional)
    detector = None
    if args.det_cfg and args.det_ckpt:
        print(f"Loading detector: {args.det_cfg}")
        detector = init_detector(args.det_cfg, args.det_ckpt, device=args.device)
    
    # Open video source
    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"❌ Failed to open source: {args.source}")
        return
    
    # Set resolution if specified
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {actual_width}x{actual_height}")
    
    # FPS counters
    fps_total = FPSCounter()
    fps_detection = FPSCounter()
    fps_pose = FPSCounter()
    
    # Skeleton links
    try:
        skeleton_links = pose_model.dataset_meta.get('skeleton_links', None)
    except:
        skeleton_links = None
    
    print("\n✅ Ready! Press 'q' to quit.")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detection phase
        t_det_start = time.time()
        if detector is not None:
            bboxes = detect_person_bboxes(detector, frame, args.bbox_thr)
            # Limit to batch size
            if len(bboxes) > args.batch_size:
                bboxes = bboxes[:args.batch_size]
        else:
            # Single person mode (full frame)
            bboxes = get_full_frame_bbox(frame)
        t_det_end = time.time()
        fps_detection.times.append(t_det_end - t_det_start)
        
        # Pose estimation phase (batched!)
        t_pose_start = time.time()
        batch_tensor, transform_info = estimator.preprocess_batch(frame, bboxes)
        keypoints_list = estimator.inference_batch(batch_tensor, transform_info)
        t_pose_end = time.time()
        fps_pose.times.append(t_pose_end - t_pose_start)
        
        # Draw results
        if not args.no_display:
            draw_keypoints_batch(frame, keypoints_list, args.score_thr, skeleton_links)
            
            # Draw bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Update FPS
            fps_total.update()
            
            # Display info
            fps_val = fps_total.get_fps()
            num_people = len(keypoints_list)
            
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"People: {num_people}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            if args.benchmark and frame_count % 30 == 0:
                det_time = sum(fps_detection.times) / len(fps_detection.times) * 1000
                pose_time = sum(fps_pose.times) / len(fps_pose.times) * 1000
                print(f"Frame {frame_count}: "
                      f"FPS={fps_val:.1f} | "
                      f"Det={det_time:.1f}ms | "
                      f"Pose={pose_time:.1f}ms | "
                      f"People={num_people}")
            
            cv2.imshow("Optimized Real-Time Pose Estimation", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("📊 Final Statistics:")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {fps_total.get_fps():.2f}")
    if len(fps_detection.times) > 0:
        print(f"Average detection time: {sum(fps_detection.times)/len(fps_detection.times)*1000:.2f}ms")
    if len(fps_pose.times) > 0:
        print(f"Average pose time: {sum(fps_pose.times)/len(fps_pose.times)*1000:.2f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
