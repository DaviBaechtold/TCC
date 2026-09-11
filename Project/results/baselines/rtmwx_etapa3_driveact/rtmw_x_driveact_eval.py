ann_file = 'driveact_midlevel.chunks_90.split_0.val.json'
codec = dict(
    decode_visibility=True,
    input_size=(
        288,
        384,
    ),
    normalize=False,
    sigma=(
        6.0,
        6.93,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_imports = dict(
    allow_failed_imports=True,
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.necks.cspnext_pafpn',
        'mmpose.models.heads.coord_cls_heads.rtmw_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.evaluation.normalized_keypoint_error',
    ])
data_root = 'data/processed/driveact/'
default_hooks = dict(logger=dict(interval=50, type='LoggerHook'))
default_scope = 'mmpose'
img_prefix = 'val2017/'
input_size = (
    288,
    384,
)
load_from = 'work_dirs/rtmw_x_driveact_ft/best_torso_px_mean_epoch_10_merged.pth'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.33,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        type='CSPNeXt',
        widen_factor=1.25),
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
            decode_visibility=True,
            input_size=(
                288,
                384,
            ),
            normalize=False,
            sigma=(
                6.0,
                6.93,
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
        in_channels=1280,
        in_featuremap_size=(
            9,
            12,
        ),
        input_size=(
            288,
            384,
        ),
        loss=dict(
            beta=1.0,
            label_beta=10.0,
            label_softmax=True,
            mask=[
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
            ],
            mask_weight=0.5,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=133,
        simcc_split_ratio=2.0,
        type='RTMWHead'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            320,
            640,
            1280,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=2,
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        type='CSPNeXtPAFPN'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
num_keypoints = 133
optim_wrapper = None
param_scheduler = None
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='driveact_midlevel.chunks_90.split_0.val.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='data/processed/driveact/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                288,
                384,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoWholeBodyDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file=
        'data/processed/driveact/driveact_midlevel.chunks_90.split_0.val.json',
        type='CocoWholeBodyMetric'),
    dict(type='TorsoNormalizedError'),
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        288,
        384,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
train_cfg = None
train_dataloader = None
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='driveact_midlevel.chunks_90.split_0.val.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='data/processed/driveact/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                288,
                384,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoWholeBodyDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file=
        'data/processed/driveact/driveact_midlevel.chunks_90.split_0.val.json',
        type='CocoWholeBodyMetric'),
    dict(type='TorsoNormalizedError'),
]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=[])
work_dir = 'results/baselines/rtmwx_etapa3_driveact'
