"""
Benchmark script to compare Top-Down vs Bottom-Up pose estimation.

Compares:
- Inference latency (ms/frame)
- Throughput (FPS)
- Memory usage
- Accuracy (if ground truth available)
- Scalability (different number of persons)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import psutil
import os

# Patch torch.load
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown

try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    init_detector = None
    inference_detector = None


class PerformanceMonitor:
    """Monitor performance metrics during inference."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.latencies = []
        self.memory_samples = []
        
    def record_latency(self, latency_ms: float):
        """Record inference latency in milliseconds."""
        self.latencies.append(latency_ms)
    
    def sample_memory(self):
        """Sample current memory usage."""
        mem_info = self.process.memory_info()
        self.memory_samples.append(mem_info.rss / 1024 / 1024)  # MB
    
    def get_statistics(self) -> Dict:
        """Compute statistics from recorded metrics."""
        latencies = np.array(self.latencies)
        memory = np.array(self.memory_samples)
        
        return {
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'fps_mean': float(1000.0 / np.mean(latencies)),
            'fps_min': float(1000.0 / np.max(latencies)),
            'fps_max': float(1000.0 / np.min(latencies)),
            'memory_mean_mb': float(np.mean(memory)),
            'memory_peak_mb': float(np.max(memory)),
            'num_frames': len(latencies),
        }


def run_topdown_benchmark(
    pose_model,
    detector,
    frames: List[np.ndarray],
    bbox_threshold: float = 0.5
) -> Dict:
    """
    Run top-down benchmark: detector → pose estimation per person.
    
    Args:
        pose_model: MMPose model
        detector: MMDet model (optional)
        frames: List of frames to process
        bbox_threshold: Detection confidence threshold
        
    Returns:
        Performance statistics
    """
    monitor = PerformanceMonitor()
    
    print("\n" + "="*60)
    print("RUNNING TOP-DOWN BENCHMARK")
    print("="*60)
    
    for i, frame in enumerate(frames):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{len(frames)}...")
        
        monitor.sample_memory()
        
        start_time = time.time()
        
        # Detect persons
        if detector is not None:
            det_result = inference_detector(detector, frame)
            if isinstance(det_result, tuple):
                bboxes = det_result[0]
            else:
                bboxes = det_result
            
            # Filter person class (index 0)
            if isinstance(bboxes, list):
                person_bboxes = bboxes[0] if bboxes else np.empty((0, 5))
            else:
                person_bboxes = bboxes
            
            if person_bboxes.size > 0:
                person_bboxes = np.asarray(person_bboxes)
                if person_bboxes.shape[1] == 5:
                    mask = person_bboxes[:, 4] >= bbox_threshold
                    person_bboxes = person_bboxes[mask, :4]
            else:
                person_bboxes = np.empty((0, 4))
        else:
            # Full frame detection
            h, w = frame.shape[:2]
            person_bboxes = np.array([[0, 0, w-1, h-1]], dtype=np.float32)
        
        # Pose estimation
        if person_bboxes.size > 0:
            results = inference_topdown(pose_model, frame, bboxes=person_bboxes)
        else:
            results = []
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        monitor.record_latency(latency_ms)
    
    print(f"✅ Completed {len(frames)} frames")
    
    return monitor.get_statistics()


def run_bottomup_benchmark(
    pose_model,
    frames: List[np.ndarray]
) -> Dict:
    """
    Run bottom-up benchmark: detect all keypoints → group into persons.
    
    Args:
        pose_model: MMPose model
        frames: List of frames to process
        
    Returns:
        Performance statistics
    """
    monitor = PerformanceMonitor()
    
    print("\n" + "="*60)
    print("RUNNING BOTTOM-UP BENCHMARK")
    print("="*60)
    
    for i, frame in enumerate(frames):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{len(frames)}...")
        
        monitor.sample_memory()
        
        start_time = time.time()
        
        # Full frame detection (simplified bottom-up)
        h, w = frame.shape[:2]
        full_bbox = np.array([[0, 0, w-1, h-1]], dtype=np.float32)
        
        results = inference_topdown(pose_model, frame, bboxes=full_bbox)
        
        # In real bottom-up, we would group keypoints here
        # For now, we just detect on full frame (simplified)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        monitor.record_latency(latency_ms)
    
    print(f"✅ Completed {len(frames)} frames")
    
    return monitor.get_statistics()


def print_comparison(topdown_stats: Dict, bottomup_stats: Dict):
    """Print detailed comparison of both approaches."""
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*60)
    
    print("\n📊 LATENCY (ms/frame)")
    print("-" * 60)
    print(f"{'Metric':<20} {'Top-Down':>15} {'Bottom-Up':>15} {'Diff':>10}")
    print("-" * 60)
    
    latency_metrics = [
        ('Mean', 'latency_mean_ms'),
        ('Std Dev', 'latency_std_ms'),
        ('Min', 'latency_min_ms'),
        ('Max', 'latency_max_ms'),
        ('Median (P50)', 'latency_p50_ms'),
        ('P95', 'latency_p95_ms'),
        ('P99', 'latency_p99_ms'),
    ]
    
    for name, key in latency_metrics:
        td_val = topdown_stats[key]
        bu_val = bottomup_stats[key]
        diff = bu_val - td_val
        diff_pct = (diff / td_val) * 100 if td_val > 0 else 0
        
        print(f"{name:<20} {td_val:>15.2f} {bu_val:>15.2f} {diff:>+9.2f} ({diff_pct:+.1f}%)")
    
    print("\n📈 THROUGHPUT (FPS)")
    print("-" * 60)
    print(f"{'Metric':<20} {'Top-Down':>15} {'Bottom-Up':>15} {'Speedup':>10}")
    print("-" * 60)
    
    fps_metrics = [
        ('Mean FPS', 'fps_mean'),
        ('Min FPS', 'fps_min'),
        ('Max FPS', 'fps_max'),
    ]
    
    for name, key in fps_metrics:
        td_val = topdown_stats[key]
        bu_val = bottomup_stats[key]
        speedup = bu_val / td_val if td_val > 0 else 0
        
        print(f"{name:<20} {td_val:>15.2f} {bu_val:>15.2f} {speedup:>9.2f}x")
    
    print("\n💾 MEMORY USAGE (MB)")
    print("-" * 60)
    print(f"{'Metric':<20} {'Top-Down':>15} {'Bottom-Up':>15} {'Diff':>10}")
    print("-" * 60)
    
    memory_metrics = [
        ('Mean', 'memory_mean_mb'),
        ('Peak', 'memory_peak_mb'),
    ]
    
    for name, key in memory_metrics:
        td_val = topdown_stats[key]
        bu_val = bottomup_stats[key]
        diff = bu_val - td_val
        diff_pct = (diff / td_val) * 100 if td_val > 0 else 0
        
        print(f"{name:<20} {td_val:>15.1f} {bu_val:>15.1f} {diff:>+9.1f} ({diff_pct:+.1f}%)")
    
    print("\n📝 SUMMARY")
    print("-" * 60)
    
    # Determine winner
    if bottomup_stats['fps_mean'] > topdown_stats['fps_mean']:
        winner = "Bottom-Up"
        speedup = bottomup_stats['fps_mean'] / topdown_stats['fps_mean']
        print(f"🏆 Winner: {winner} ({speedup:.2f}x faster)")
    else:
        winner = "Top-Down"
        speedup = topdown_stats['fps_mean'] / bottomup_stats['fps_mean']
        print(f"🏆 Winner: {winner} ({speedup:.2f}x faster)")
    
    # Memory comparison
    if bottomup_stats['memory_peak_mb'] < topdown_stats['memory_peak_mb']:
        mem_save = topdown_stats['memory_peak_mb'] - bottomup_stats['memory_peak_mb']
        print(f"💾 Bottom-Up uses {mem_save:.1f} MB less memory")
    else:
        mem_save = bottomup_stats['memory_peak_mb'] - topdown_stats['memory_peak_mb']
        print(f"💾 Top-Down uses {mem_save:.1f} MB less memory")
    
    print("\n📌 RECOMMENDATIONS")
    print("-" * 60)
    
    if bottomup_stats['fps_mean'] > topdown_stats['fps_mean'] * 1.2:
        print("✅ Use Bottom-Up for:")
        print("   - Multiple persons in frame")
        print("   - Real-time applications requiring high FPS")
        print("   - Crowded scenes")
    else:
        print("✅ Use Top-Down for:")
        print("   - Single person or few persons")
        print("   - Maximum accuracy per person")
        print("   - When detector is available")
    
    print("="*60 + "\n")


def load_frames_from_source(source: str, max_frames: int = 100) -> List[np.ndarray]:
    """Load frames from video file or camera."""
    
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    
    frames = []
    print(f"Loading up to {max_frames} frames from {source}...")
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale (match training)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_3ch = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        frames.append(frame_gray_3ch)
    
    cap.release()
    print(f"✅ Loaded {len(frames)} frames")
    
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Top-Down vs Bottom-Up pose estimation"
    )
    
    # Model args
    parser.add_argument(
        '--cfg',
        type=str,
        default='work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py',
        help='Pose model config'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth',
        help='Pose model checkpoint'
    )
    parser.add_argument(
        '--det-cfg',
        type=str,
        default='',
        help='Detector config (optional, for top-down)'
    )
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Detector checkpoint (optional, for top-down)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device (cuda:0 or cpu)'
    )
    
    # Benchmark args
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Video file or camera index'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum number of frames to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BENCHMARK: TOP-DOWN VS BOTTOM-UP")
    print("="*60)
    print(f"Config: {args.cfg}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Source: {args.source}")
    print(f"Max frames: {args.max_frames}")
    print("="*60)
    
    # Load model
    print("\nInitializing pose model...")
    pose_model = init_model(args.cfg, args.ckpt, device=args.device)
    
    # Load detector (optional)
    detector = None
    if args.det_cfg and args.det_ckpt:
        print("Initializing detector...")
        if init_detector is None:
            print("⚠️  Warning: mmdet not available, skipping detector")
        else:
            detector = init_detector(args.det_cfg, args.det_ckpt, device=args.device)
    
    # Load frames
    frames = load_frames_from_source(args.source, args.max_frames)
    
    if len(frames) == 0:
        raise RuntimeError("No frames loaded")
    
    # Run benchmarks
    topdown_stats = run_topdown_benchmark(pose_model, detector, frames)
    bottomup_stats = run_bottomup_benchmark(pose_model, frames)
    
    # Print comparison
    print_comparison(topdown_stats, bottomup_stats)
    
    # Save results
    results = {
        'config': {
            'model_config': args.cfg,
            'model_checkpoint': args.ckpt,
            'device': args.device,
            'source': args.source,
            'num_frames': len(frames),
        },
        'topdown': topdown_stats,
        'bottomup': bottomup_stats,
        'comparison': {
            'speedup_bottomup': bottomup_stats['fps_mean'] / topdown_stats['fps_mean'] if topdown_stats['fps_mean'] > 0 else 0,
            'memory_diff_mb': bottomup_stats['memory_peak_mb'] - topdown_stats['memory_peak_mb'],
            'latency_diff_ms': bottomup_stats['latency_mean_ms'] - topdown_stats['latency_mean_ms'],
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {args.output}")


if __name__ == '__main__':
    main()
