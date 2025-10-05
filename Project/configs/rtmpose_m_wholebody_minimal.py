# Minimal RTMPose configuration for troubleshooting
# This version bypasses problematic components

"""
Minimal RTMPose configuration for troubleshooting
- Avoids importing mmpose.visualization (which triggers mmcv.ops)
- Imports only the required mmpose submodules explicitly
"""

default_scope = 'mmpose'

custom_imports = dict(
    imports=[
        # Import only the modules needed to register what we use
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.heads.rtmcc_head',
        # Dataset + metrics
        'mmpose.datasets',
        'mmpose.evaluation',
    ],
    allow_failed_imports=True,
)

# Model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=133,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(
        flip_test=True,
        shift_heatmap=False,
        align_corners=False))

# Dataset settings
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/processed/grayscale/'

# Simple pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]

# Data loaders with reduced batch size for safety
train_dataloader = dict(
    batch_size=32,  # Reduced from 80
    num_workers=4,  # Reduced from 10
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=32,  # Reduced from 64
    num_workers=4,  # Reduced from 8
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# Evaluators
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'annotations/coco_wholebody_val_v1.0.json')

# Training settings
train_cfg = dict(
    by_epoch=True,
    max_epochs=50,  # Reduced for testing
    val_interval=5)

val_cfg = dict()

# Optimization
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=2e-3, weight_decay=0.05))  # Removed AMP for now

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=2e-5,
        by_epoch=True,
        begin=10,
        end=50)
]

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco-wholebody/AP',
        rule='greater',
        max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=10))

# Simplified logging
log_processor = dict(
    type='LogProcessor',
    window_size=10,
    by_epoch=True)

# Minimal visualizer - no 3D components
# Use mmengine's simple Visualizer to avoid importing mmpose.visualization
visualizer = dict(type='Visualizer')