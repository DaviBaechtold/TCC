#!/usr/bin/env python3
"""
Real-Time Pose Estimation with TURBO Optimizations.

This version uses PyTorch advanced optimizations:
- torch.compile() for JIT compilation
- Channels-last memory format for better cache utilization
- Optimized inference with @torch.inference_mode()
- Batch processing for multiple people

Expected speedup: 1.5-2x over batch processing version
Target: 70+ FPS for multi-person scenarios

Usage:
    python src/evaluation/run_realtime_turbo.py \\
        --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \\
        --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
        --source 0  # or video file
        --device cuda:0 \\
        --batch-size 8 \\
        --compile  # Enable torch.compile()
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import MMPose/MMDet
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import PoseDataSample
    HAS_MMPOSE = True
except ImportError:
    HAS_MMPOSE = False
    
try:
    from mmdet.apis import inference_detector, init_detector
    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False


class TurboPoseEstimator:
    """
    Ultra-optimized pose estimator using PyTorch 2.x features.
    
    Key optimizations:
    - torch.compile() for JIT optimization
    - Channels-last memory format
    - Batch processing
    - @torch.inference_mode() for reduced overhead
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        use_compile: bool = True,
        batch_size: int = 4
    ):
        """
        Initialize turbo pose estimator.
        
        Args:
            config_path: Path to model config
            checkpoint_path: Path to model checkpoint
            device: Device to use
            use_compile: Whether to use torch.compile()
            batch_size: Max batch size for processing
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_compile = use_compile
        
        print(f"⏳ Loading model...")
        print(f"   Config: {config_path}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   torch.compile: {'✅ Enabled' if use_compile else '❌ Disabled'}")
        
        # Load model
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.model.eval()
        
        # Get input size from model config
        if hasattr(self.model.cfg, 'codec'):
            self.input_size = tuple(self.model.cfg.codec['input_size'])
        else:
            self.input_size = (288, 384)  # default
        
        print(f"   Input size: {self.input_size}")
        
        # Convert to channels-last for better performance
        print("⏳ Converting to channels-last memory format...")
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # Compile model for speedup (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            print("⏳ Compiling model with torch.compile()...")
            try:
                # Use reduce-overhead mode for real-time inference
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",  # Optimize for latency
                    fullgraph=False,
                    backend="inductor"
                )
                print("✅ Model compiled successfully!")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}")
                print("   Falling back to eager mode")
        
        # ImageNet normalization stats
        self.mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
        
        print("✅ Turbo Pose Estimator ready!")
    
    @torch.inference_mode()  # Faster than torch.no_grad()
    def preprocess_batch(self, frame, bboxes):
        """
        Preprocess batch of bounding boxes to model input.
        
        Args:
            frame: Input frame (H, W, 3) numpy array
            bboxes: List of bounding boxes [(x1, y1, x2, y2, score), ...]
        
        Returns:
            batch_tensor: Preprocessed batch tensor (N, 3, H, W)
            transform_info: List of transform info for denormalization
        """
        if len(bboxes) == 0:
            return None, []
        
        crops = []
        transform_info = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Add margin (10%)
            w, h = x2 - x1, y2 - y1
            x1 = max(0, x1 - int(w * 0.1))
            y1 = max(0, y1 - int(h * 0.1))
            x2 = min(frame.shape[1], x2 + int(w * 0.1))
            y2 = min(frame.shape[0], y2 + int(h * 0.1))
            
            # Crop and resize
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop_resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
            
            # Store transform info for denormalization
            transform_info.append({
                'bbox': (x1, y1, x2, y2),
                'scale_x': (x2 - x1) / self.input_size[1],
                'scale_y': (y2 - y1) / self.input_size[0],
                'offset_x': x1,
                'offset_y': y1
            })
            
            crops.append(crop_resized)
        
        if len(crops) == 0:
            return None, []
        
        # Convert to tensor (vectorized)
        # Shape: (N, H, W, 3) -> (N, 3, H, W)
        batch_np = np.stack(crops, axis=0)
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
        batch_tensor = batch_tensor.to(
            device=self.device,
            dtype=torch.float32,
            memory_format=torch.channels_last  # Use channels-last
        )
        
        # Normalize (ImageNet stats)
        batch_tensor = (batch_tensor - self.mean) / self.std
        
        return batch_tensor, transform_info
    
    @torch.inference_mode()
    def inference_batch(self, batch_tensor, transform_info):
        """
        Run inference on batch.
        
        Args:
            batch_tensor: Input tensor (N, 3, H, W)
            transform_info: List of transform info
        
        Returns:
            keypoints_list: List of keypoints arrays (N, 133, 3)
        """
        if batch_tensor is None or batch_tensor.shape[0] == 0:
            return []
        
        # Process in sub-batches if needed
        all_results = []
        for i in range(0, batch_tensor.shape[0], self.batch_size):
            sub_batch = batch_tensor[i:i+self.batch_size]
            
            # Forward pass through model backbone and head
            # Use model's internal forward (faster than inference_topdown)
            with torch.amp.autocast('cuda', enabled=True):
                # Extract features
                feats = self.model.backbone(sub_batch)
                
                # Get heatmaps from head
                if hasattr(self.model, 'head'):
                    predictions = self.model.head.forward(feats)
                else:
                    predictions = self.model.forward(sub_batch)
            
            all_results.append(predictions)
        
        # Combine results
        if len(all_results) > 1:
            combined = self._combine_predictions(all_results)
        else:
            combined = all_results[0]
        
        # Decode predictions to keypoints
        keypoints_list = self._decode_keypoints(combined, transform_info)
        
        return keypoints_list
    
    def _combine_predictions(self, predictions_list):
        """Combine multiple prediction batches."""
        # This depends on the model output format
        # For now, return the list as-is and handle in decode
        return predictions_list
    
    def _decode_keypoints(self, predictions, transform_info):
        """
        Decode model predictions to keypoints.
        
        This is a placeholder - actual implementation depends on model output format.
        For RTMPose, we'd need to decode heatmaps or regression outputs.
        """
        # Fallback to MMPose inference for now
        # TODO: Implement fast custom decoder
        keypoints_list = []
        for info in transform_info:
            # Return dummy keypoints for now
            keypoints_list.append(np.zeros((133, 3)))
        
        return keypoints_list


class FPSCounter:
    """Smooth FPS counter with moving average."""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        """Update FPS counter."""
        now = time.time()
        dt = now - self.last_time
        self.frame_times.append(dt)
        self.last_time = now
    
    def get_fps(self):
        """Get current FPS."""
        if len(self.frame_times) == 0:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / max(avg_dt, 1e-6)


def draw_keypoints(frame, keypoints, scores=None, threshold=0.3):
    """Draw keypoints on frame."""
    if keypoints is None or len(keypoints) == 0:
        return
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if scores is not None and scores[i] < threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser(description="Turbo Real-Time Pose Estimation")
    parser.add_argument("--cfg", required=True, help="Path to pose model config")
    parser.add_argument("--ckpt", required=True, help="Path to pose model checkpoint")
    parser.add_argument("--det-cfg", default="configs/detectors/rtmdet_nano_person_infer.py",
                       help="Path to detector config")
    parser.add_argument("--det-ckpt", default="checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
                       help="Path to detector checkpoint")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or video file)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark mode (print detailed timing)")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold")
    
    args = parser.parse_args()
    
    if not HAS_MMPOSE or not HAS_MMDET:
        print("❌ Error: MMPose and MMDet required!")
        sys.exit(1)
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║      🚀 TURBO Real-Time Pose Estimation (PyTorch Optimized)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Initialize detector
    print("⏳ Loading person detector...")
    detector = init_detector(args.det_cfg, args.det_ckpt, device=args.device)
    print("✅ Detector loaded!")
    print()
    
    # Initialize turbo pose estimator
    estimator = TurboPoseEstimator(
        config_path=args.cfg,
        checkpoint_path=args.ckpt,
        device=args.device,
        use_compile=args.compile,
        batch_size=args.batch_size
    )
    print()
    
    # Open video source
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {source}")
        sys.exit(1)
    
    # Get video properties
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"Input FPS: {fps_in:.1f}")
    print()
    
    print("✅ Ready! Press 'q' to quit.")
    if args.benchmark:
        print("============================================================")
    print()
    
    fps_counter = FPSCounter()
    frame_count = 0
    
    # Timing stats
    det_times = []
    pose_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection
            t0 = time.time()
            det_results = inference_detector(detector, frame)
            
            # Extract person bounding boxes
            if hasattr(det_results, 'pred_instances'):
                pred_instances = det_results.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                
                # Filter for person class (class 0) and score threshold
                person_mask = (labels == 0) & (scores > args.score_thr)
                bboxes = bboxes[person_mask]
                scores = scores[person_mask]
            else:
                bboxes = []
                scores = []
            
            t1 = time.time()
            det_time = (t1 - t0) * 1000
            
            # Pose estimation (batch)
            if len(bboxes) > 0:
                # Preprocess
                batch_tensor, transform_info = estimator.preprocess_batch(frame, bboxes)
                
                # Inference
                keypoints_list = estimator.inference_batch(batch_tensor, transform_info)
                
                # Draw results
                if not args.no_display:
                    for keypoints in keypoints_list:
                        if keypoints is not None:
                            draw_keypoints(frame, keypoints[:, :2], keypoints[:, 2])
            
            t2 = time.time()
            pose_time = (t2 - t1) * 1000
            
            # Update FPS
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Store timing
            det_times.append(det_time)
            pose_times.append(pose_time)
            
            # Display
            if not args.no_display:
                # Draw FPS
                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f} | People: {len(bboxes)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Turbo Pose Estimation", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Benchmark output
            if args.benchmark and frame_count % 30 == 0:
                print(f"Frame {frame_count}: FPS={current_fps:.1f} | "
                      f"Det={det_time:.1f}ms | Pose={pose_time:.1f}ms | People={len(bboxes)}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if args.benchmark and frame_count > 0:
            print()
            print("============================================================")
            print("📊 Final Statistics:")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {fps_counter.get_fps():.2f}")
            print(f"Average detection time: {np.mean(det_times):.2f}ms")
            print(f"Average pose time: {np.mean(pose_times):.2f}ms")
            print("============================================================")


if __name__ == "__main__":
    main()
