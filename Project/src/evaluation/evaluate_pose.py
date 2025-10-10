"""Run RTMPose inference on RGB and IR images and compare results."""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

# Register safe globals for torch.load
for typ in [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.str_]:
    torch.serialization.add_safe_globals([typ])

# Monkeypatch torch.load
_original_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(f, *args, **kwargs)
torch.load = _patched_load

try:
    from mmpose.apis import init_model, inference_topdown
except ImportError as e:
    print(f"Error: {e}")
    print("Please install mmpose")
    sys.exit(1)


def extract_kps(result):
    """Extract keypoints from PoseDataSample."""
    if not result or len(result) == 0:
        return None
    
    pred_inst = result[0]
    
    if hasattr(pred_inst, 'pred_instances'):
        instances = pred_inst.pred_instances
        if hasattr(instances, 'keypoints'):
            keypoints = instances.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            else:
                keypoints = np.asarray(keypoints)
            return keypoints
    
    return None


def normalized_keypoint_distance(kp1, kp2):
    """Compute normalized distance between two keypoint sets."""
    if kp1 is None or kp2 is None:
        return float('inf')
    
    if kp1.ndim == 3:
        kp1 = kp1[0]
    if kp2.ndim == 3:
        kp2 = kp2[0]
    
    # Normalize by torso distance (shoulders: keypoints 5-6)
    torso_dist = np.linalg.norm(kp1[5, :2] - kp1[6, :2])
    if torso_dist < 1e-6:
        torso_dist = 1.0
    
    dists = np.linalg.norm(kp1[:, :2] - kp2[:, :2], axis=1)
    mean_dist = np.mean(dists)
    
    return mean_dist / torso_dist


def main():
    parser = argparse.ArgumentParser(description='Evaluate RTMPose on RGB vs IR')
    parser.add_argument('--cfg', type=str, 
                        default='work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py')
    parser.add_argument('--ckpt', type=str,
                        default='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth')
    parser.add_argument('--rgb-dir', type=str, default='data/raw/val2017')
    parser.add_argument('--ir-dir', type=str, default='data/processed/grayscale/val2017')
    parser.add_argument('--out-dir', type=str, default='work_dirs/eval_results')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.out_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'ir'), exist_ok=True)
    
    print(f"Loading model from {args.ckpt}...")
    
    # Initialize model
    model = init_model(args.cfg, args.ckpt, device=args.device)
    print("Model loaded successfully!")
    
    # Get image list
    rgb_images = sorted([f for f in os.listdir(args.rgb_dir) if f.endswith(('.jpg', '.png'))])
    rgb_images = rgb_images[:args.n]
    
    print(f"Processing {len(rgb_images)} images...")
    
    distances = []
    
    for i, img_name in enumerate(rgb_images):
        print(f"[{i+1}/{len(rgb_images)}] Processing {img_name}...")
        
        rgb_path = os.path.join(args.rgb_dir, img_name)
        ir_path = os.path.join(args.ir_dir, img_name)
        
        if not os.path.exists(ir_path):
            print(f"  Warning: IR image not found: {ir_path}")
            continue
        
        # Read images
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        if rgb_img is None or ir_img is None:
            print(f"  Warning: Could not read images")
            continue
        
        h, w = rgb_img.shape[:2]
        
        # Use full-image bounding box in xyxy format
        bbox = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)
        
        # Run inference with manual bbox
        try:
            rgb_result = inference_topdown(model, rgb_img, bboxes=bbox)
            ir_result = inference_topdown(model, ir_img, bboxes=bbox)
        except Exception as e:
            print(f"  Error during inference: {e}")
            continue
        
        # Extract keypoints
        rgb_kps = extract_kps(rgb_result)
        ir_kps = extract_kps(ir_result)
        
        # Compute distance
        if rgb_kps is not None and ir_kps is not None:
            dist = normalized_keypoint_distance(rgb_kps, ir_kps)
            distances.append(dist)
            print(f"  Normalized distance: {dist:.4f}")
        else:
            print(f"  Warning: Could not extract keypoints")
        
        # Visualize and save
        try:
            # Draw keypoints on RGB
            rgb_vis = rgb_img.copy()
            if rgb_kps is not None:
                kp = rgb_kps[0] if rgb_kps.ndim == 3 else rgb_kps
                for j, (x, y) in enumerate(kp[:, :2]):
                    cv2.circle(rgb_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(args.out_dir, 'rgb', img_name), rgb_vis)
            
            # Draw keypoints on IR
            ir_vis = ir_img.copy()
            if ir_kps is not None:
                kp = ir_kps[0] if ir_kps.ndim == 3 else ir_kps
                for j, (x, y) in enumerate(kp[:, :2]):
                    cv2.circle(ir_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(args.out_dir, 'ir', img_name), ir_vis)
            
        except Exception as e:
            print(f"  Warning: Visualization error: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(distances)}")
    if distances:
        print(f"Mean normalized distance: {np.mean(distances):.4f}")
        print(f"Median normalized distance: {np.median(distances):.4f}")
        print(f"Std normalized distance: {np.std(distances):.4f}")
        print(f"Min distance: {np.min(distances):.4f}")
        print(f"Max distance: {np.max(distances):.4f}")
    else:
        print("No valid distance measurements obtained")
    print(f"\nVisualizations saved to:")
    print(f"  RGB: {os.path.join(args.out_dir, 'rgb')}")
    print(f"  IR:  {os.path.join(args.out_dir, 'ir')}")
    print("="*60)


if __name__ == '__main__':
    main()
