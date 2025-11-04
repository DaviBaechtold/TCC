#!/usr/bin/env python3
"""
Debug script to test bottom-up implementation and diagnose shape issues.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch

# Patch torch.load
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown


def test_inference_shapes(config_path: str, checkpoint_path: str):
    """Test inference and print all intermediate shapes."""
    
    print("=" * 60)
    print("BOTTOM-UP INFERENCE SHAPE DEBUG")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Loading model...")
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    print(f"   ✅ Model loaded")
    
    # Create dummy frame
    print("\n2. Creating test frame (640x480)...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    print(f"   Frame shape: {frame.shape}")
    
    # Create full-image bbox
    print("\n3. Creating full-image bbox...")
    full_bbox = np.array([[0, 0, w, h, 1.0]])
    print(f"   Bbox shape: {full_bbox.shape}")
    print(f"   Bbox: {full_bbox}")
    
    # Run inference
    print("\n4. Running inference...")
    try:
        results = inference_topdown(model, frame, bboxes=full_bbox)
        print(f"   ✅ Inference successful")
        print(f"   Number of results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n   Result {i}:")
            
            if not hasattr(result, 'pred_instances'):
                print("      ❌ No pred_instances")
                continue
            
            instances = result.pred_instances
            print(f"      ✅ Has pred_instances")
            
            # Check keypoints
            if hasattr(instances, 'keypoints'):
                kpts = instances.keypoints
                
                if hasattr(kpts, 'cpu'):
                    kpts_np = kpts.cpu().numpy()
                else:
                    kpts_np = np.asarray(kpts)
                
                print(f"      Keypoints shape: {kpts_np.shape}")
                print(f"      Keypoints dtype: {kpts_np.dtype}")
                print(f"      Keypoints ndim: {kpts_np.ndim}")
                
                # Show sample keypoint
                if kpts_np.size > 0:
                    if kpts_np.ndim == 3:
                        print(f"      First keypoint: {kpts_np[0, 0]}")
                    elif kpts_np.ndim == 2:
                        print(f"      First keypoint: {kpts_np[0]}")
            else:
                print("      ❌ No keypoints")
            
            # Check keypoint_scores
            if hasattr(instances, 'keypoint_scores'):
                scores = instances.keypoint_scores
                
                if hasattr(scores, 'cpu'):
                    scores_np = scores.cpu().numpy()
                else:
                    scores_np = np.asarray(scores)
                
                print(f"      Keypoint_scores shape: {scores_np.shape}")
                print(f"      Keypoint_scores dtype: {scores_np.dtype}")
                print(f"      Keypoint_scores ndim: {scores_np.ndim}")
                
                # Show sample score
                if scores_np.size > 0:
                    if scores_np.ndim == 2:
                        print(f"      First score: {scores_np[0, 0]}")
                    elif scores_np.ndim == 1:
                        print(f"      First score: {scores_np[0]}")
            else:
                print("      ⚠️  No keypoint_scores")
            
            # Check bboxes
            if hasattr(instances, 'bboxes'):
                bboxes = instances.bboxes
                if hasattr(bboxes, 'cpu'):
                    bboxes_np = bboxes.cpu().numpy()
                else:
                    bboxes_np = np.asarray(bboxes)
                print(f"      Bboxes shape: {bboxes_np.shape}")
            else:
                print("      ⚠️  No bboxes")
        
        print("\n" + "=" * 60)
        print("SHAPE COMPATIBILITY CHECK")
        print("=" * 60)
        
        # Check if we can concatenate
        for i, result in enumerate(results):
            if not hasattr(result, 'pred_instances'):
                continue
            
            instances = result.pred_instances
            
            if not hasattr(instances, 'keypoints'):
                continue
            
            kpts = instances.keypoints
            if hasattr(kpts, 'cpu'):
                kpts_np = kpts.cpu().numpy()
            else:
                kpts_np = np.asarray(kpts)
            
            # Add batch dimension if needed
            if kpts_np.ndim == 2:
                kpts_np = kpts_np[None, ...]
            
            print(f"\nResult {i}:")
            print(f"  Keypoints shape after reshaping: {kpts_np.shape}")
            
            if kpts_np.shape[-1] == 2:
                print(f"  Need to add confidence dimension")
                
                if hasattr(instances, 'keypoint_scores'):
                    scores = instances.keypoint_scores
                    if hasattr(scores, 'cpu'):
                        scores_np = scores.cpu().numpy()
                    else:
                        scores_np = np.asarray(scores)
                    
                    if scores_np.ndim == 1:
                        scores_np = scores_np[None, :]
                    
                    print(f"  Scores shape: {scores_np.shape}")
                    print(f"  Scores [..., None] shape: {scores_np[..., None].shape}")
                    
                    # Check compatibility
                    if (scores_np.shape[0] == kpts_np.shape[0] and 
                        scores_np.shape[1] == kpts_np.shape[1]):
                        print(f"  ✅ Shapes compatible for concatenation")
                        
                        # Try concatenation
                        try:
                            result_kpts = np.concatenate(
                                [kpts_np, scores_np[..., None]],
                                axis=-1
                            )
                            print(f"  ✅ Concatenation successful: {result_kpts.shape}")
                        except Exception as e:
                            print(f"  ❌ Concatenation failed: {e}")
                    else:
                        print(f"  ❌ Shapes NOT compatible")
                        print(f"     kpts:  ({kpts_np.shape[0]}, {kpts_np.shape[1]}, 2)")
                        print(f"     scores: ({scores_np.shape[0]}, {scores_np.shape[1]})")
                else:
                    print(f"  Using default confidence")
                    conf = np.ones((kpts_np.shape[0], kpts_np.shape[1], 1)) * 0.8
                    print(f"  Confidence shape: {conf.shape}")
                    
                    try:
                        result_kpts = np.concatenate([kpts_np, conf], axis=-1)
                        print(f"  ✅ Concatenation successful: {result_kpts.shape}")
                    except Exception as e:
                        print(f"  ❌ Concatenation failed: {e}")
            else:
                print(f"  ✅ Already has confidence (shape[-1] = {kpts_np.shape[-1]})")
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    config = "work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py"
    checkpoint = "work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth"
    
    test_inference_shapes(config, checkpoint)
