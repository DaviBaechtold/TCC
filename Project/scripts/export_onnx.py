#!/usr/bin/env python3
"""
Export MMPose models to ONNX format for faster inference.

This script exports RTMPose models to ONNX, which can provide
significant speedup (2-3x) compared to PyTorch inference.

Usage:
    python scripts/export_onnx.py --model pose
    python scripts/export_onnx.py --model detector
    python scripts/export_onnx.py --export-all
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mmpose.apis import init_model
    from mmdet.apis import init_detector
except ImportError as e:
    print(f"❌ Error importing MMPose/MMDet: {e}")
    sys.exit(1)


def print_header():
    """Print script header."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         🚀 ONNX Export for Real-Time Pose Estimation            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()


def export_pose_to_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple = (1, 3, 288, 384),
    opset_version: int = 11,
    device: str = "cuda:0"
):
    """
    Export RTMPose model to ONNX format.
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version
        device: Device to use for export
    """
    print("============================================================")
    print("📦 Exporting RTMPose Model to ONNX")
    print("============================================================")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input shape: {input_shape}")
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load model
        print("⏳ Loading model...")
        model = init_model(config_path, checkpoint_path, device=device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Export to ONNX
        print("⏳ Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        # Verify export
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ Successfully exported to: {output_path}")
        
        # Print model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 Model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_detector_to_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    opset_version: int = 11,
    device: str = "cuda:0"
):
    """
    Export RTMDet model to ONNX format.
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version
        device: Device to use for export
    """
    print("============================================================")
    print("📦 Exporting RTMDet Detector to ONNX")
    print("============================================================")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input shape: {input_shape}")
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load model
        print("⏳ Loading model...")
        model = init_detector(config_path, checkpoint_path, device=device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Export to ONNX
        print("⏳ Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        # Verify export
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ Successfully exported to: {output_path}")
        
        # Print model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 Model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model", choices=["pose", "detector", "all"], 
                       help="Which model to export")
    parser.add_argument("--export-all", action="store_true",
                       help="Export all models")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use for export")
    parser.add_argument("--opset", type=int, default=11,
                       help="ONNX opset version")
    
    args = parser.parse_args()
    
    print_header()
    
    # Check for onnx
    try:
        import onnx
        print(f"✅ ONNX version: {onnx.__version__}")
    except ImportError:
        print("❌ ONNX not installed!")
        print("   Install: pip install onnx onnxruntime-gpu")
        sys.exit(1)
    
    print()
    
    results = []
    
    # Export based on arguments
    if args.model == "detector" or args.export_all:
        success = export_detector_to_onnx(
            config_path="configs/detectors/rtmdet_nano_person_infer.py",
            checkpoint_path="checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
            output_path="deploy/onnx/rtmdet_nano_person.onnx",
            input_shape=(1, 3, 640, 640),
            opset_version=args.opset,
            device=args.device
        )
        results.append(("RTMDet", success))
        print()
    
    if args.model == "pose" or args.export_all:
        success = export_pose_to_onnx(
            config_path="work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py",
            checkpoint_path="work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth",
            output_path="deploy/onnx/rtmpose_m_wholebody.onnx",
            input_shape=(1, 3, 288, 384),
            opset_version=args.opset,
            device=args.device
        )
        results.append(("RTMPose", success))
        print()
    
    # Print summary
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                      📊 EXPORT SUMMARY                           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    for model_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    print()
    print("📝 Next steps:")
    print("   1. Test ONNX models with ONNXRuntime")
    print("   2. (Optional) Convert ONNX to TensorRT with trtexec")
    print("   3. Benchmark performance improvement")
    print("   4. Expected speedup: 2-3x with ONNX, 3-5x with TensorRT")
    print()


if __name__ == "__main__":
    main()
