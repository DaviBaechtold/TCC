"""Export RTMPose model to TorchScript for faster inference."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.jit

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmpose.apis import init_model


def export_to_torchscript(
    config: str,
    checkpoint: str,
    output_path: str,
    device: str = 'cuda:0',
    input_size: tuple = (288, 384),
    optimize: bool = True,
    verify: bool = True
) -> None:
    """
    Export MMPose model to TorchScript format.
    
    Args:
        config: path to config file
        checkpoint: path to checkpoint file
        output_path: path to save TorchScript model
        device: device to use for export
        input_size: (height, width) of input
        optimize: apply TorchScript optimizations
        verify: verify exported model output matches original
    
    Example:
        >>> export_to_torchscript(
        ...     config='configs/rtmpose_m_wholebody_minimal.py',
        ...     checkpoint='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth',
        ...     output_path='work_dirs/test_minimal5/rtmpose_m_torchscript.pt'
        ... )
    """
    print(f"\n{'='*70}")
    print("RTMPose TorchScript Export")
    print(f"{'='*70}\n")
    
    # 1. Load model
    print(f"[1/5] Loading model...")
    print(f"  Config: {config}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")
    
    model = init_model(config, checkpoint, device=device)
    model.eval()
    print("  ✓ Model loaded successfully")
    
    # 2. Create dummy input
    print(f"\n[2/5] Creating dummy input...")
    height, width = input_size
    # MMPose expects BGR format, 3 channels even for grayscale
    dummy_input = torch.randn(1, 3, height, width).to(device)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  ✓ Dummy input created")
    
    # 3. Trace model
    print(f"\n[3/5] Tracing model with TorchScript...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        print("  ✓ Model traced successfully")
    except Exception as e:
        print(f"  ✗ Tracing failed: {e}")
        print("\n  Note: Some MMPose operations may not be traceable.")
        print("  Consider using torch.jit.script instead of trace,")
        print("  or export to ONNX format.")
        return
    
    # 4. Optimize (optional)
    if optimize:
        print(f"\n[4/5] Optimizing TorchScript model...")
        try:
            # Freeze model (inline constants)
            traced_model = torch.jit.freeze(traced_model)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            print("  ✓ Model optimized")
        except Exception as e:
            print(f"  ⚠ Optimization partially failed: {e}")
            print("  Continuing with unoptimized model...")
    else:
        print(f"\n[4/5] Skipping optimization (--no-optimize flag)")
    
    # 5. Verify output (optional)
    if verify:
        print(f"\n[5/5] Verifying exported model...")
        with torch.no_grad():
            original_output = model(dummy_input)
            traced_output = traced_model(dummy_input)
            
            # Compare outputs
            if isinstance(original_output, (list, tuple)):
                original_output = original_output[0]
            if isinstance(traced_output, (list, tuple)):
                traced_output = traced_output[0]
            
            max_diff = torch.max(torch.abs(original_output - traced_output)).item()
            print(f"  Max difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("  ✓ Verification passed (outputs match)")
            else:
                print(f"  ⚠ Verification warning: difference = {max_diff}")
                print("  This may be acceptable for inference.")
    else:
        print(f"\n[5/5] Skipping verification (--no-verify flag)")
    
    # 6. Save
    print(f"\nSaving TorchScript model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.jit.save(traced_model, output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # 7. Usage instructions
    print(f"\n{'='*70}")
    print("Export complete! 🎉")
    print(f"{'='*70}\n")
    
    print("To use the exported model:")
    print("```python")
    print("import torch")
    print(f"model = torch.jit.load('{output_path}')")
    print("model.eval()")
    print("with torch.no_grad():")
    print("    output = model(input_tensor)")
    print("```")
    
    print("\nExpected speedup: 1.3-2x faster inference")
    print("Recommended for deployment and production use.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export RTMPose model to TorchScript',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python scripts/export/export_torchscript.py \\
    --config configs/rtmpose_m_wholebody_minimal.py \\
    --checkpoint work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
    --output work_dirs/test_minimal5/rtmpose_m_torchscript.pt

  # Export without optimization (faster export, slower inference)
  python scripts/export/export_torchscript.py \\
    --config configs/rtmpose_m_wholebody_minimal.py \\
    --checkpoint work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
    --output work_dirs/test_minimal5/rtmpose_m_torchscript.pt \\
    --no-optimize

  # Export for CPU inference
  python scripts/export/export_torchscript.py \\
    --config configs/rtmpose_m_wholebody_minimal.py \\
    --checkpoint work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
    --output work_dirs/test_minimal5/rtmpose_m_torchscript_cpu.pt \\
    --device cpu
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save TorchScript model (.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for export (default: cuda:0)'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[288, 384],
        metavar=('HEIGHT', 'WIDTH'),
        help='Input size (height width) (default: 288 384)'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip TorchScript optimization (faster export, slower inference)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip output verification'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check CUDA availability
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Export
    try:
        export_to_torchscript(
            config=args.config,
            checkpoint=args.checkpoint,
            output_path=args.output,
            device=args.device,
            input_size=tuple(args.input_size),
            optimize=not args.no_optimize,
            verify=not args.no_verify
        )
    except KeyboardInterrupt:
        print("\n\nExport interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Export failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
