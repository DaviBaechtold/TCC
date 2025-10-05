#!/usr/bin/env python3
"""
Test training script with environment workarounds for MMCV ops issues.
This script attempts to work around the missing compiled extensions.
"""

import os
import sys
import warnings

# Suppress MMCV extension warnings
os.environ['MMCV_WITH_OPS'] = '0'  # Disable MMCV compiled ops
warnings.filterwarnings('ignore', category=UserWarning)

# Try to patch the problematic module before importing
try:
    import mmcv.ops
    print("❌ MMCV ops imported - this might cause issues")
except Exception as e:
    print("✅ MMCV ops import failed as expected:", str(e)[:50] + "...")

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    try:
        # Import and run the original training script
        from src.training.train_pose import main as train_main
        print("🚀 Starting training with ultra-minimal config...")
        
        # Override sys.argv to use our ultra-minimal config
        original_argv = sys.argv.copy()
        sys.argv = [
            'test_train_minimal.py',
            '--config', 'configs/rtmpose_m_wholebody_ultra_minimal.py',
            '--load-from', 'checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
            '--work-dir', 'work_dirs/test_ultra_minimal'
        ]
        
        train_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This indicates a fundamental issue with the environment setup.")
        return 1
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("This might be due to MMCV ops or other configuration issues.")
        return 1
        
    finally:
        # Restore original argv
        if 'original_argv' in locals():
            sys.argv = original_argv

    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)