# Config de avaliação isolada do RTMPose-m WholeBody (256x192).
#
# Serve para medir AP/AR de QUALQUER checkpoint compatível sobre QUALQUER
# diretório de imagens, trocando apenas `data_root`/`img_prefix` via
# --cfg-options. Usado para quantificar o domain gap RGB -> grayscale.

default_scope = 'mmpose'

custom_imports = dict(
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.heads.rtmcc_head',
        'mmpose.datasets',
        'mmpose.evaluation',
    ],
    allow_failed_imports=True)

# Trocáveis por --cfg-options
data_root = 'data/processed/grayscale/'
img_prefix = 'val2017/'
ann_file = 'annotations/coco_wholebody_val_v1.0.json'

input_size = (192, 256)  # (w, h)

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

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
        input_size=input_size,
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
        decoder=codec),
    test_cfg=dict(flip_test=True, shift_heatmap=False, align_corners=False))

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

test_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file=ann_file,
        data_prefix=dict(img=img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
    ))

val_dataloader = test_dataloader

test_evaluator = dict(
    type='CocoWholeBodyMetric', ann_file=data_root + ann_file)
val_evaluator = test_evaluator

test_cfg = dict()
val_cfg = dict()
train_cfg = None
train_dataloader = None
optim_wrapper = None
param_scheduler = None

default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=False)
visualizer = dict(type='PoseLocalVisualizer', vis_backends=[])
work_dir = 'work_dirs/eval_zero_shot'
