"""RTMPose-s configuration tuned for grayscale (infrared-style) whole-body pose.

This variant targets real-time inference by using the small RTMPose backbone
with 256x192 input resolution and aggressive AMP-enabled training.
"""

default_scope = 'mmpose'

# Optional lightweight imports to register required modules when running
custom_imports = dict(
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.heads.rtmcc_head',
        'mmpose.datasets',
        'mmpose.evaluation',
    ],
    allow_failed_imports=True,
)

# Dataset statistics (estimated on grayscale train split)
_GRAY_MEAN = 109.6
_GRAY_STD = 56.2

data_root = 'data/processed/grayscale/'
data_mode = 'topdown'
dataset_type = 'CocoWholeBodyDataset'

codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[_GRAY_MEAN, _GRAY_MEAN, _GRAY_MEAN],
        std=[_GRAY_STD, _GRAY_STD, _GRAY_STD],
        bgr_to_rgb=False),  # images are already grayscale replicated across channels
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'),
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=133,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
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
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion', brightness_delta=10, contrast_range=(0.8, 1.2)),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.05),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
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
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=6,
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
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'annotations/coco_wholebody_val_v1.0.json',
)

test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=3e-3, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=3e-5,
        by_epoch=True,
        begin=60,
        end=210,
    ),
]

auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco-wholebody/AP',
        rule='greater',
        max_keep_ckpts=2,
    ),
    logger=dict(type='LoggerHook', interval=50),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
]

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)
