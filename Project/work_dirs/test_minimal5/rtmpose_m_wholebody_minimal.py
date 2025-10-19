custom_imports = dict(
    allow_failed_imports=True,
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.heads.rtmcc_head',
        'mmpose.datasets',
        'mmpose.evaluation',
    ])
data_mode = 'topdown'
data_root = 'data/processed/grayscale/'
dataset_type = 'CocoWholeBodyDataset'
default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco-wholebody/AP',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'))
default_scope = 'mmpose'
gpu_ids = [
    0,
]
load_from = 'checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=133,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(align_corners=False, flip_test=True, shift_heatmap=False),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.002, type='AdamW', weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        begin=10,
        by_epoch=True,
        end=50,
        eta_min=2e-05,
        type='CosineAnnealingLR'),
]
seed = 42
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=5)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_mode='topdown',
        data_prefix=dict(img='train2017/'),
        data_root='data/processed/grayscale/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoWholeBodyDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='data/processed/grayscale/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoWholeBodyDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'data/processed/grayscale/annotations/coco_wholebody_val_v1.0.json',
    type='CocoWholeBodyMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='data/processed/grayscale/',
        pipeline=test_pipeline,
        test_mode=True,
        type='CocoWholeBodyDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
visualizer = dict(type='Visualizer')
work_dir = 'work_dirs/test_minimal5'
