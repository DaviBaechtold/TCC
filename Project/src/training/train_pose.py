"""
Script de treinamento para RTMPose em imagens grayscale.

Este script implementa o treinamento do modelo RTMPose para estimação
de pose full-body em imagens grayscale (simulação de infrared).
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RTMPose for grayscale images')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rtmpose_m_wholebody.py',
        help='Config file path'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Working directory to save logs and checkpoints'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='Load weights from checkpoint (for fine-tuning)'
    )
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=[0],
        help='GPU ids to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set work directory
    if args.work_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = f'work_dirs/rtmpose_grayscale_{timestamp}'
    
    cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        cfg.seed = args.seed
    
    # Set deterministic
    if args.deterministic:
        cfg.deterministic = True
    
    # Set GPU ids
    cfg.gpu_ids = args.gpu_ids
    
    # Set AMP
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    
    # Set resume and load_from
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
    elif args.load_from:
        cfg.load_from = args.load_from
    
    # Print config
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Config file:    {args.config}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"GPU IDs:        {cfg.gpu_ids}")
    print(f"Seed:           {cfg.seed}")
    print(f"AMP:            {args.amp}")
    print("=" * 80)
    
    # Build runner
    print("\n🚀 Initializing runner...")
    runner = Runner.from_cfg(cfg)
    
    # Print model info
    print("\n📊 Model Information:")
    print(f"Model type: {cfg.model.type}")
    print(f"Backbone:   {cfg.model.backbone.type}")
    print(f"Head:       {cfg.model.head.type}")
    print(f"Keypoints:  {cfg.model.head.out_channels}")
    
    # Print dataset info
    print("\n📚 Dataset Information:")
    train_dataset = cfg.train_dataloader.dataset
    val_dataset = cfg.val_dataloader.dataset
    print(f"Dataset type:       {train_dataset.type}")
    print(f"Training samples:   {len(train_dataset) if hasattr(train_dataset, '__len__') else 'N/A'}")
    print(f"Validation samples: {len(val_dataset) if hasattr(val_dataset, '__len__') else 'N/A'}")
    print(f"Batch size:         {cfg.train_dataloader.batch_size}")
    print(f"Num workers:        {cfg.train_dataloader.num_workers}")
    
    # Print training info
    print("\n⚙️  Training Configuration:")
    print(f"Max epochs:      {cfg.train_cfg.max_epochs}")
    print(f"Val interval:    {cfg.train_cfg.val_interval}")
    print(f"Optimizer:       {cfg.optim_wrapper.optimizer.type}")
    print(f"Learning rate:   {cfg.optim_wrapper.optimizer.lr}")
    print(f"Weight decay:    {cfg.optim_wrapper.optimizer.weight_decay}")
    
    print("\n" + "=" * 80)
    print("🎯 Starting Training...")
    print("=" * 80 + "\n")
    
    # Start training
    try:
        runner.train()
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Training failed with error: {e}")
        print("=" * 80)
        raise
    
    finally:
        # Print final info
        print("\n📁 Output files:")
        print(f"Checkpoints: {cfg.work_dir}")
        print(f"Logs:        {cfg.work_dir}/*/")
        print(f"Config:      {cfg.work_dir}/*.py")


if __name__ == '__main__':
    main()
