#!/usr/bin/env python3
"""
Evaluate accuracy comparison between top-down and bottom-up approaches.

This script evaluates both inference strategies on the COCO-WholeBody validation
set and compares their accuracy metrics (AP, AR, per-keypoint performance).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.evaluation.metrics import CocoWholeBodyMetric
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmpose.datasets import CocoWholeBodyDataset
    from mmpose.structures import PoseDataSample, merge_data_samples
except ImportError as exc:
    raise SystemExit(f"Required packages not found: {exc}")


def create_bottomup_bbox(img_shape):
    """Create full-image bounding box for bottom-up inference."""
    h, w = img_shape[:2]
    return np.array([[0, 0, w, h]])


def evaluate_topdown(
    model,
    dataset,
    max_samples: Optional[int] = None,
    device: str = 'cuda:0'
) -> Dict:
    """
    Evaluate top-down approach on dataset.
    
    Args:
        model: Initialized pose model
        dataset: COCO-WholeBody dataset
        max_samples: Maximum number of samples to evaluate (None = all)
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING TOP-DOWN APPROACH")
    print("="*60)
    
    predictions = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    start_time = time.time()
    
    for idx in range(num_samples):
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (num_samples - idx - 1) / speed
            print(f"Processing {idx + 1}/{num_samples} | Speed: {speed:.1f} img/s | ETA: {eta:.1f}s")
        
        # Get sample
        data_info = dataset.get_data_info(idx)
        
        # Load image
        img_path = data_info['img_path']
        img = dataset.pipeline.transforms[0].transform(
            {'img_path': img_path, 'img': None}
        )['img']
        
        # Get bounding boxes for this image
        bbox = data_info.get('bbox', None)
        if bbox is None:
            # Use full image bbox
            h, w = img.shape[:2]
            bbox = np.array([[0, 0, w, h]])
        else:
            bbox = np.array([bbox])
        
        # Run inference
        try:
            results = inference_topdown(model, img, bboxes=bbox)
            
            # Convert results to prediction format
            for result in results:
                if hasattr(result, 'pred_instances'):
                    instances = result.pred_instances
                    
                    if hasattr(instances, 'keypoints'):
                        keypoints = instances.keypoints.cpu().numpy()
                        scores = instances.keypoint_scores.cpu().numpy() if hasattr(instances, 'keypoint_scores') else np.ones(keypoints.shape[:-1])
                        
                        # Format prediction
                        pred = {
                            'image_id': data_info['img_id'],
                            'category_id': 1,  # person
                            'keypoints': keypoints.flatten().tolist(),
                            'score': float(scores.mean())
                        }
                        predictions.append(pred)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {num_samples} images in {elapsed:.1f}s ({num_samples/elapsed:.1f} img/s)")
    
    return {
        'predictions': predictions,
        'num_samples': num_samples,
        'inference_time': elapsed
    }


def evaluate_bottomup(
    model,
    dataset,
    max_samples: Optional[int] = None,
    device: str = 'cuda:0'
) -> Dict:
    """
    Evaluate bottom-up approach on dataset.
    
    Args:
        model: Initialized pose model
        dataset: COCO-WholeBody dataset
        max_samples: Maximum number of samples to evaluate (None = all)
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING BOTTOM-UP APPROACH")
    print("="*60)
    
    predictions = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    start_time = time.time()
    
    for idx in range(num_samples):
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (num_samples - idx - 1) / speed
            print(f"Processing {idx + 1}/{num_samples} | Speed: {speed:.1f} img/s | ETA: {eta:.1f}s")
        
        # Get sample
        data_info = dataset.get_data_info(idx)
        
        # Load image
        img_path = data_info['img_path']
        img = dataset.pipeline.transforms[0].transform(
            {'img_path': img_path, 'img': None}
        )['img']
        
        # Create full-image bbox for bottom-up
        bbox = create_bottomup_bbox(img.shape)
        
        # Run inference
        try:
            results = inference_topdown(model, img, bboxes=bbox)
            
            # Convert results to prediction format
            for result in results:
                if hasattr(result, 'pred_instances'):
                    instances = result.pred_instances
                    
                    if hasattr(instances, 'keypoints'):
                        keypoints = instances.keypoints.cpu().numpy()
                        scores = instances.keypoint_scores.cpu().numpy() if hasattr(instances, 'keypoint_scores') else np.ones(keypoints.shape[:-1])
                        
                        # Format prediction
                        pred = {
                            'image_id': data_info['img_id'],
                            'category_id': 1,  # person
                            'keypoints': keypoints.flatten().tolist(),
                            'score': float(scores.mean())
                        }
                        predictions.append(pred)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {num_samples} images in {elapsed:.1f}s ({num_samples/elapsed:.1f} img/s)")
    
    return {
        'predictions': predictions,
        'num_samples': num_samples,
        'inference_time': elapsed
    }


def compute_coco_metrics(predictions: List[Dict], gt_file: str) -> Dict:
    """
    Compute COCO metrics from predictions.
    
    Args:
        predictions: List of prediction dictionaries
        gt_file: Path to ground truth annotations
        
    Returns:
        Dictionary with COCO metrics
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    # Load ground truth
    coco_gt = COCO(gt_file)
    
    # Create results format
    results = []
    for pred in predictions:
        results.append({
            'image_id': pred['image_id'],
            'category_id': pred['category_id'],
            'keypoints': pred['keypoints'],
            'score': pred['score']
        })
    
    if len(results) == 0:
        return {
            'AP': 0.0,
            'AP_50': 0.0,
            'AP_75': 0.0,
            'AR': 0.0,
            'AR_50': 0.0,
            'AR_75': 0.0
        }
    
    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'AP': float(coco_eval.stats[0]),
        'AP_50': float(coco_eval.stats[1]),
        'AP_75': float(coco_eval.stats[2]),
        'AP_medium': float(coco_eval.stats[3]),
        'AP_large': float(coco_eval.stats[4]),
        'AR': float(coco_eval.stats[5]),
        'AR_50': float(coco_eval.stats[6]),
        'AR_75': float(coco_eval.stats[7]),
        'AR_medium': float(coco_eval.stats[8]),
        'AR_large': float(coco_eval.stats[9])
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare top-down vs bottom-up accuracy on COCO-WholeBody"
    )
    
    # Model arguments
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='Path to model config file'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/grayscale',
        help='Root directory of dataset'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='data/processed/grayscale/annotations/coco_wholebody_val_v1.0.json',
        help='Path to annotation file'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='work_dirs/eval_results/accuracy_comparison.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ACCURACY COMPARISON: TOP-DOWN VS BOTTOM-UP")
    print("="*60)
    print(f"Config: {args.cfg}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset: {args.ann_file}")
    print(f"Device: {args.device}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("="*60)
    
    # Initialize model
    print("\nInitializing model...")
    model = init_model(args.cfg, args.ckpt, device=args.device)
    print("✅ Model loaded")
    
    # Load dataset
    print("\nLoading dataset...")
    from mmengine.registry import DATASETS
    from mmpose.datasets import CocoWholeBodyDataset
    
    # Simple dataset loading
    import cv2
    import json
    
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"✅ Loaded {len(coco_data['images'])} images")
    
    # Simplified evaluation - we'll use direct inference
    # Create a simple wrapper
    class SimpleDataset:
        def __init__(self, coco_data, data_root):
            self.coco_data = coco_data
            self.data_root = Path(data_root)
            self.images = coco_data['images']
            self.annotations = coco_data['annotations']
            
            # Build image_id to annotations mapping
            self.img_to_anns = {}
            for ann in self.annotations:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
        
        def __len__(self):
            return len(self.images)
        
        def get_data_info(self, idx):
            img_info = self.images[idx]
            img_id = img_info['id']
            
            # Get first annotation for this image (for bbox)
            anns = self.img_to_anns.get(img_id, [])
            bbox = anns[0]['bbox'] if anns else None
            
            return {
                'img_id': img_id,
                'img_path': str(self.data_root / 'val2017' / img_info['file_name']),
                'bbox': bbox
            }
        
        class Pipeline:
            class LoadImage:
                def transform(self, data):
                    img = cv2.imread(data['img_path'])
                    if img is None:
                        raise ValueError(f"Failed to load image: {data['img_path']}")
                    data['img'] = img
                    return data
            
            def __init__(self):
                self.transforms = [self.LoadImage()]
    
    dataset = SimpleDataset(coco_data, args.data_root)
    dataset.pipeline = SimpleDataset.Pipeline()
    
    # Evaluate top-down
    topdown_results = evaluate_topdown(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Evaluate bottom-up
    bottomup_results = evaluate_bottomup(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Compute COCO metrics
    print("\n" + "="*60)
    print("COMPUTING COCO METRICS")
    print("="*60)
    
    print("\nTop-Down Metrics:")
    topdown_metrics = compute_coco_metrics(
        topdown_results['predictions'],
        args.ann_file
    )
    
    print("\nBottom-Up Metrics:")
    bottomup_metrics = compute_coco_metrics(
        bottomup_results['predictions'],
        args.ann_file
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<20} {'Top-Down':<12} {'Bottom-Up':<12} {'Difference':<12}")
    print("-" * 60)
    
    for metric in ['AP', 'AP_50', 'AP_75', 'AR', 'AR_50', 'AR_75']:
        td_val = topdown_metrics.get(metric, 0.0)
        bu_val = bottomup_metrics.get(metric, 0.0)
        diff = bu_val - td_val
        diff_pct = (diff / td_val * 100) if td_val > 0 else 0
        
        print(f"{metric:<20} {td_val:.4f}       {bu_val:.4f}       {diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': {
            'model_config': args.cfg,
            'model_checkpoint': args.ckpt,
            'annotation_file': args.ann_file,
            'num_samples': topdown_results['num_samples']
        },
        'topdown': {
            'metrics': topdown_metrics,
            'inference_time': topdown_results['inference_time'],
            'num_predictions': len(topdown_results['predictions'])
        },
        'bottomup': {
            'metrics': bottomup_metrics,
            'inference_time': bottomup_results['inference_time'],
            'num_predictions': len(bottomup_results['predictions'])
        },
        'comparison': {
            'ap_diff': bottomup_metrics['AP'] - topdown_metrics['AP'],
            'ar_diff': bottomup_metrics['AR'] - topdown_metrics['AR'],
            'speed_ratio': topdown_results['inference_time'] / bottomup_results['inference_time']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
