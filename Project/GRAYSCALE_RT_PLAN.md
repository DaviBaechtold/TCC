# Real-time Grayscale Full-Body Pose Estimation Plan

This document tracks the action plan for training a RTMPose model on the grayscale (infrared-style) dataset.

## Objectives
- Train a lightweight RTMPose variant that can run in real time (≥30 FPS) on consumer GPUs.
- Specialize the model on the grayscale WholeBody split (`data/processed/grayscale`).
- Produce checkpoints and evaluation metrics tailored to IR imagery.

## Model Strategy
- **Backbone**: CSPNeXt-P5 configured for the small RTMPose variant (`deepen_factor=0.33`, `widen_factor=0.5`).
- **Input resolution**: 256×192 (height×width) to balance accuracy and speed.
- **Head**: RTMCCHead with SimCC decoder (out_channels=133 for WholeBody).
- **Precision**: Mixed precision (AMP) enabled to maximize throughput during training and inference.
- **Initialization**: Start from COCO WholeBody pretrained weights if available; otherwise, train from scratch with warmup.

## Data Pipeline
- **Dataset**: `CocoWholeBodyDataset` pointing at `data/processed/grayscale/` (train/val splits already prepared).
- **Transforms**:
  - Load 3-channel grayscale images (already replicated per channel).
  - Standard top-down augmentations: random flip, half-body, scale/rotation jitter.
  - Light photometric/blur jitter (kept modest to preserve IR characteristics).
  - SimCC target generation.
- **Normalization**: Use dataset-specific statistics (mean≈109.6, std≈56.2) instead of ImageNet defaults.

## Training Schedule
- **Epochs**: 210 (validate every 10 epochs).
- **Batch size**: 64 (adjustable based on GPU memory).
- **Optimizer**: AdamW, lr=3e-3 with cosine decay (eta_min=3e-5).
- **Warmup**: Linear warmup for first 1k iterations.
- **Hooks**: EMA hook for stability, checkpoint best on WholeBody AP.

## Evaluation & Deployment
- **Metrics**: `CocoWholeBodyMetric` on grayscale validation split.
- **Artifacts**: Store checkpoints/metrics under `work_dirs/grayscale_rt/`.
- **Real-time validation**: Benchmark on GPU with batch_size=1 using `src/evaluation/evaluate_pose.py` after training.

## Action Items
1. Create dedicated config (`configs/rtmpose_s_grayscale_rt.py`) reflecting the settings above.
2. Update training script usage docs (`README`, `QUICKSTART`) to point to the new config.
3. Launch training via `scripts/train_full_pipeline.sh` with the new config and AMP enabled.
4. Monitor logs, adjust LR/batch size if instability occurs, and collect validation curves.
5. After convergence, profile inference speed and compare against real-time target.
