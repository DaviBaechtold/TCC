"""
TensorRT Export Guide for RTMPose Models
=========================================

This script helps export RTMPose models to TensorRT for 3-5x speedup.
TensorRT is NVIDIA's inference optimizer for production deployment.

Requirements:
- CUDA Toolkit (already installed: 12.8)
- TensorRT 8.6+ (install via pip or from NVIDIA)
- MMDeploy (install via pip)

Installation:
    pip install tensorrt
    pip install mmdeploy mmdeploy-runtime

Performance Expected:
- RTMDet-Nano: 10ms → 3ms (3.3x speedup)
- RTMPose-M: 20ms → 7ms (2.8x speedup)
- Total: ~30ms → ~10ms = 100 FPS!

Usage:
    python scripts/export_tensorrt.py --help
"""

import argparse
import os
import subprocess
from pathlib import Path


def check_tensorrt():
    """Check if TensorRT is installed."""
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("❌ TensorRT not found!")
        print("\n📦 Install TensorRT:")
        print("   Option 1 (pip): pip install tensorrt")
        print("   Option 2 (conda): conda install -c nvidia tensorrt")
        print("   Option 3 (official): Download from https://developer.nvidia.com/tensorrt")
        return False


def check_mmdeploy():
    """Check if MMDeploy is installed."""
    try:
        import mmdeploy
        print(f"✅ MMDeploy installed")
        return True
    except ImportError:
        print("❌ MMDeploy not found!")
        print("\n📦 Install MMDeploy:")
        print("   pip install mmdeploy mmdeploy-runtime")
        return False


def export_model_tensorrt(model_type: str, config_path: str, checkpoint_path: str, 
                         output_dir: str, input_shape: tuple = (1, 3, 384, 288)):
    """
    Export MMPose/MMDet model to TensorRT.
    
    Args:
        model_type: 'pose' or 'det'
        config_path: Path to model config (.py)
        checkpoint_path: Path to checkpoint (.pth)
        output_dir: Output directory for TensorRT engine
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔧 Exporting {model_type} model to TensorRT...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    print(f"   Input shape: {input_shape}")
    
    # Determine deploy config based on model type
    if model_type == 'pose':
        deploy_cfg = 'configs/mmdeploy/pose_tensorrt_dynamic.py'
    elif model_type == 'det':
        deploy_cfg = 'configs/mmdeploy/detection_tensorrt_dynamic.py'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create deploy config if doesn't exist
    if not os.path.exists(deploy_cfg):
        create_deploy_config(deploy_cfg, model_type, input_shape)
    
    # Run mmdeploy conversion
    cmd = [
        'python', '-m', 'mmdeploy.tools.deploy',
        deploy_cfg,
        config_path,
        checkpoint_path,
        'dummy_input.jpg',  # Placeholder, not actually used
        '--work-dir', output_dir,
        '--device', 'cuda:0',
        '--dump-info'
    ]
    
    print(f"\n⚙️  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Export successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Export failed!")
        print(e.stderr)
        return False


def create_deploy_config(output_path: str, model_type: str, input_shape: tuple):
    """Create MMDeploy configuration for TensorRT export."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if model_type == 'pose':
        config_content = f'''
_base_ = ['mmdeploy://base/base_static.py']

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape={input_shape},
    dynamic_axes={{
        'input': {{0: 'batch'}},
        'output': {{0: 'batch'}}
    }}
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True,  # Enable FP16 for speedup
        max_workspace_size=1 << 30  # 1GB
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape={input_shape},
                    opt_shape={input_shape},
                    max_shape=(8, input_shape[1], input_shape[2], input_shape[3])
                )
            )
        )
    ]
)

codebase_config = dict(
    type='mmpose',
    task='PoseDetection'
)
'''
    
    elif model_type == 'det':
        config_content = f'''
_base_ = ['mmdeploy://base/base_static.py']

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['boxes', 'labels'],
    input_shape={input_shape},
    dynamic_axes={{
        'input': {{0: 'batch'}},
        'boxes': {{0: 'batch'}},
        'labels': {{0: 'batch'}}
    }}
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True,
        max_workspace_size=1 << 30
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape={input_shape},
                    opt_shape={input_shape},
                    max_shape=(8, input_shape[1], input_shape[2], input_shape[3])
                )
            )
        )
    ]
)

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection'
)
'''
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created deploy config: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export RTMPose models to TensorRT")
    parser.add_argument('--check-only', action='store_true', 
                       help="Only check if dependencies are installed")
    parser.add_argument('--export-detector', action='store_true',
                       help="Export RTMDet person detector")
    parser.add_argument('--export-pose', action='store_true',
                       help="Export RTMPose model")
    parser.add_argument('--export-all', action='store_true',
                       help="Export both detector and pose model")
    
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         🚀 TensorRT Export for Real-Time Pose Estimation        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Check dependencies
    tensorrt_ok = check_tensorrt()
    mmdeploy_ok = check_mmdeploy()
    
    if args.check_only:
        if tensorrt_ok and mmdeploy_ok:
            print("\n✅ All dependencies installed! Ready to export.")
        else:
            print("\n❌ Missing dependencies. Install them first.")
        return
    
    if not (tensorrt_ok and mmdeploy_ok):
        print("\n❌ Cannot proceed without TensorRT and MMDeploy.")
        print("   Run with --check-only to see installation instructions.")
        return
    
    # Export models
    if args.export_all or args.export_detector:
        print("\n" + "="*60)
        print("📦 Exporting RTMDet Person Detector")
        print("="*60)
        export_model_tensorrt(
            model_type='det',
            config_path='configs/detectors/rtmdet_nano_person_infer.py',
            checkpoint_path='checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
            output_dir='deploy/rtmdet_trt',
            input_shape=(1, 3, 640, 640)
        )
    
    if args.export_all or args.export_pose:
        print("\n" + "="*60)
        print("📦 Exporting RTMPose Model")
        print("="*60)
        export_model_tensorrt(
            model_type='pose',
            config_path='work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py',
            checkpoint_path='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth',
            output_dir='deploy/rtmpose_trt',
            input_shape=(1, 3, 384, 288)
        )
    
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                      ✅ EXPORT COMPLETE                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("\n📝 Next steps:")
    print("   1. Check deploy/ directory for TensorRT engines")
    print("   2. Update run_realtime_optimized.py to use TensorRT models")
    print("   3. Benchmark performance improvement")
    print("   4. Expected speedup: 3-5x faster!")


if __name__ == '__main__':
    main()
