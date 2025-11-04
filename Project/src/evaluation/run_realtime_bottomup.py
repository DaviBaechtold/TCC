"""
Bottom-Up Pose Estimation with Automatic Bounding Box Detection

This implements a simplified bottom-up approach for real-time pose estimation:
1. Detect all keypoints in the image without person detection
2. Group keypoints into individual persons using spatial proximity
3. Generate bounding boxes automatically from grouped keypoints

Advantages over top-down:
- Faster for multiple persons (no detector overhead)
- Single forward pass per frame
- Better for crowded scenes

Disadvantages:
- Slightly lower accuracy per person
- More complex grouping algorithm
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import PoseDataSample
except ImportError as exc:
    raise SystemExit(f"mmpose is required: {exc}")


# COCO-WholeBody skeleton for visualization
SKELETON_LINKS = [
    # Body (17 keypoints)
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (5, 11), (6, 12),  # Torso
]

BODY_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def compute_bbox_from_keypoints(
    keypoints: np.ndarray,
    padding_ratio: float = 0.15,
    min_confidence: float = 0.3
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bounding box from a set of keypoints.
    
    Args:
        keypoints: (N, 3) array of [x, y, confidence]
        padding_ratio: Ratio of bbox size to add as padding
        min_confidence: Minimum confidence to include keypoint
        
    Returns:
        (x1, y1, x2, y2) bounding box or None if no valid keypoints
    """
    # Filter by confidence
    valid_mask = keypoints[:, 2] >= min_confidence
    valid_kpts = keypoints[valid_mask, :2]
    
    if len(valid_kpts) == 0:
        return None
    
    # Compute min/max
    x_min, y_min = valid_kpts.min(axis=0)
    x_max, y_max = valid_kpts.max(axis=0)
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = int(x_max + pad_x)
    y2 = int(y_max + pad_y)
    
    return (x1, y1, x2, y2)


def group_keypoints_by_proximity(
    all_keypoints: np.ndarray,
    max_distance: float = 100.0,
    min_keypoints: int = 5,
    confidence_threshold: float = 0.3
) -> List[np.ndarray]:
    """
    Group detected keypoints into individual persons using spatial clustering.
    
    Strategy:
    1. Use high-confidence body keypoints (shoulders, hips) as anchors
    2. Cluster these anchors to identify person centers
    3. Assign all keypoints to nearest person cluster
    4. Validate each group has minimum keypoints
    
    Args:
        all_keypoints: (M, N, 3) where M is # detections, N is # keypoints per person
        max_distance: Maximum distance to consider keypoints as belonging to same person
        min_keypoints: Minimum number of keypoints to form a valid person
        confidence_threshold: Minimum confidence for a keypoint to be used
        
    Returns:
        List of (N, 3) arrays, one per detected person
    """
    if len(all_keypoints) == 0:
        return []
    
    # Body keypoint indices (COCO-WholeBody format)
    # Focus on torso keypoints which are most reliable
    BODY_CENTER_INDICES = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip
    
    grouped_persons = []
    
    for person_keypoints in all_keypoints:
        # Extract high-confidence body center keypoints
        body_centers = []
        for idx in BODY_CENTER_INDICES:
            if idx < len(person_keypoints):
                kpt = person_keypoints[idx]
                if kpt[2] > confidence_threshold:
                    body_centers.append(kpt[:2])
        
        # Need at least 2 body keypoints to define a person
        if len(body_centers) < 2:
            # Fall back to any valid keypoints
            valid_kpts = person_keypoints[person_keypoints[:, 2] > confidence_threshold]
            if len(valid_kpts) >= min_keypoints:
                grouped_persons.append(person_keypoints)
            continue
        
        # Compute center of mass for this person
        body_centers = np.array(body_centers)
        person_center = body_centers.mean(axis=0)
        
        # Check if this person overlaps with existing groups
        is_new_person = True
        for existing_person in grouped_persons:
            # Compute existing person's center
            existing_centers = []
            for idx in BODY_CENTER_INDICES:
                if idx < len(existing_person):
                    kpt = existing_person[idx]
                    if kpt[2] > confidence_threshold:
                        existing_centers.append(kpt[:2])
            
            if len(existing_centers) >= 2:
                existing_center = np.array(existing_centers).mean(axis=0)
                distance = np.linalg.norm(person_center - existing_center)
                
                # If centers are close, merge with existing person
                if distance < max_distance:
                    is_new_person = False
                    # Keep keypoints with higher confidence
                    for i in range(len(person_keypoints)):
                        if person_keypoints[i, 2] > existing_person[i, 2]:
                            existing_person[i] = person_keypoints[i]
                    break
        
        # Add as new person if doesn't overlap
        if is_new_person:
            valid_count = (person_keypoints[:, 2] > confidence_threshold).sum()
            if valid_count >= min_keypoints:
                grouped_persons.append(person_keypoints.copy())
    
    return grouped_persons


def draw_keypoints_and_bbox(
    frame: np.ndarray,
    keypoints: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Draw keypoints, skeleton, and bounding box on frame.
    
    Args:
        frame: Input image
        keypoints: (N, 3) array of [x, y, confidence]
        bbox: Optional (x1, y1, x2, y2) bounding box
        color: RGB color for drawing
        thickness: Line thickness
        confidence_threshold: Minimum confidence to draw keypoint
        
    Returns:
        Frame with annotations
    """
    img = frame.copy()
    
    # Draw bounding box if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = "Person"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            img,
            (x1, y1 - label_size[1] - 5),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    # Draw skeleton links (body only for now)
    for link in SKELETON_LINKS:
        if link[0] < len(keypoints) and link[1] < len(keypoints):
            pt1 = keypoints[link[0]]
            pt2 = keypoints[link[1]]
            
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                cv2.line(
                    img,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    max(1, thickness // 2)
                )
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            # Use different colors for different body parts
            if i < 5:  # Head
                kp_color = (255, 0, 0)  # Blue
            elif i < 11:  # Arms
                kp_color = (0, 255, 0)  # Green
            else:  # Legs
                kp_color = (0, 0, 255)  # Red
            
            cv2.circle(img, (int(x), int(y)), 3, kp_color, -1)
            cv2.circle(img, (int(x), int(y)), 4, color, 1)
    
    return img


class BottomUpPoseEstimator:
    """
    Simplified bottom-up pose estimator using top-down model in sliding window.
    
    Note: This is a simplified implementation. True bottom-up would use
    models like HigherHRNet or AssociativeEmbedding.
    """
    
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = 'cuda:0',
        confidence_threshold: float = 0.3
    ):
        """
        Initialize bottom-up pose estimator.
        
        Args:
            config: Path to model config
            checkpoint: Path to model checkpoint
            device: Device to run inference on
            confidence_threshold: Minimum confidence for keypoint detection
        """
        self.model = init_model(config, checkpoint, device=device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        
    def __call__(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all persons in frame using bottom-up approach.
        
        Args:
            frame: Input image (H, W, 3)
            
        Returns:
            List of dicts with 'keypoints' and 'bbox' for each person
        """
        # Strategy: Use full-image inference
        # In true bottom-up, this would detect all keypoints then group them
        # Here we simplify by using the model on full image
        
        h, w = frame.shape[:2]
        
        # Create a "fake" bbox covering the whole image
        # This allows us to use the top-down model in bottom-up fashion
        # Note: MMPose expects bboxes as [x1, y1, x2, y2] without score
        full_bbox = np.array([[0, 0, w, h]])
        
        try:
            # Run inference on full image
            results = inference_topdown(self.model, frame, bboxes=full_bbox)
            
            if not results or len(results) == 0:
                return []
            
            # Extract keypoints from results
            all_persons = []
            
            for result in results:
                if not hasattr(result, 'pred_instances'):
                    continue
                
                instances = result.pred_instances
                
                if not hasattr(instances, 'keypoints'):
                    continue
                
                keypoints = instances.keypoints
                
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                else:
                    keypoints = np.asarray(keypoints)
                
                # keypoints shape: (num_instances, num_keypoints, 2 or 3)
                if keypoints.ndim == 2:
                    keypoints = keypoints[None, ...]
                
                # Add confidence scores if not present
                if keypoints.shape[-1] == 2:
                    # Estimate confidence from keypoint scores if available
                    if hasattr(instances, 'keypoint_scores'):
                        scores = instances.keypoint_scores
                        if hasattr(scores, 'cpu'):
                            scores = scores.cpu().numpy()
                        else:
                            scores = np.asarray(scores)
                        
                        # Ensure scores match keypoints shape
                        if scores.ndim == 1:
                            scores = scores[None, :]
                        
                        # scores should be (num_instances, num_keypoints)
                        # keypoints is (num_instances, num_keypoints, 2)
                        # We need to add last dimension to scores
                        if scores.shape[0] == keypoints.shape[0] and scores.shape[1] == keypoints.shape[1]:
                            keypoints = np.concatenate(
                                [keypoints, scores[..., None]],
                                axis=-1
                            )
                        else:
                            # Shape mismatch, use default confidence
                            confidence = np.ones((keypoints.shape[0], keypoints.shape[1], 1)) * 0.8
                            keypoints = np.concatenate([keypoints, confidence], axis=-1)
                    else:
                        # Use default confidence
                        confidence = np.ones((keypoints.shape[0], keypoints.shape[1], 1)) * 0.8
                        keypoints = np.concatenate([keypoints, confidence], axis=-1)
                
                # Process each detected instance
                for kpts in keypoints:
                    # Compute bounding box from keypoints
                    bbox = compute_bbox_from_keypoints(
                        kpts,
                        padding_ratio=0.15,
                        min_confidence=self.confidence_threshold
                    )
                    
                    if bbox is not None:
                        all_persons.append({
                            'keypoints': kpts,
                            'bbox': bbox
                        })
            
            return all_persons
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description="Bottom-up real-time pose estimation with auto bbox detection"
    )
    
    # Model arguments
    parser.add_argument(
        '--cfg',
        type=str,
        default='work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py',
        help='Path to pose model config'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth',
        help='Path to pose model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run inference (cuda:0 or cpu)'
    )
    
    # Input source
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Input source: webcam index (0), video file path, or image folder'
    )
    
    # Visualization
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (optional)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in window'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold for keypoint detection'
    )
    
    args = parser.parse_args()
    
    # Initialize estimator
    print(f"Initializing bottom-up pose estimator...")
    print(f"  Config: {args.cfg}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Device: {args.device}")
    
    estimator = BottomUpPoseEstimator(
        config=args.cfg,
        checkpoint=args.ckpt,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Setup input source
    if args.source.isdigit():
        # Webcam
        source = int(args.source)
        print(f"\nUsing webcam {source}")
    else:
        # Video file
        source = args.source
        if not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        print(f"\nUsing video file: {source}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    
    # Setup output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"\nSaving output to: {args.output}")
    
    # Colors for different persons
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    # Performance tracking
    frame_count = 0
    total_time = 0
    fps_display = 0
    
    print("\n" + "="*60)
    print("BOTTOM-UP POSE ESTIMATION - REAL-TIME")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  SPACE - Pause/Resume")
    print("="*60 + "\n")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video or read error")
                    break
                
                # Convert to grayscale (match training data)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray_3ch = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                
                # Run inference
                start_time = time.time()
                persons = estimator(frame_gray_3ch)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                frame_count += 1
                
                if frame_count % 10 == 0:
                    fps_display = 1.0 / (total_time / frame_count)
                
                # Draw results
                vis_frame = frame.copy()
                
                for i, person in enumerate(persons):
                    color = colors[i % len(colors)]
                    vis_frame = draw_keypoints_and_bbox(
                        vis_frame,
                        person['keypoints'],
                        person['bbox'],
                        color=color,
                        confidence_threshold=args.confidence_threshold
                    )
                
                # Add info overlay
                info_text = [
                    f"FPS: {fps_display:.1f}",
                    f"Latency: {inference_time*1000:.1f}ms",
                    f"Persons: {len(persons)}",
                    f"Frame: {frame_count}",
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        vis_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 30
                
                # Show mode indicator
                cv2.putText(
                    vis_frame,
                    "BOTTOM-UP MODE",
                    (width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )
                
                # Write to output
                if writer is not None:
                    writer.write(vis_frame)
                
                # Display
                if args.show or isinstance(args.source, int) or args.source.isdigit():
                    cv2.imshow('Bottom-Up Pose Estimation', vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f'frame_{frame_count:06d}.jpg'
                cv2.imwrite(save_path, vis_frame)
                print(f"Saved frame to {save_path}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count / total_time:.2f}")
        print(f"Average latency: {(total_time / frame_count) * 1000:.2f}ms")
        print("="*60)


if __name__ == '__main__':
    main()
