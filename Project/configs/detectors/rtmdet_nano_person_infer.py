# RTMDet-nano config for person detection inference (MMDet 3.x compatible)

default_scope = 'mmdet'

# Model settings
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.25,  # match nano checkpoint channel widths
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,  # Only person class
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Minimal dataloader definition required by init_detector in MMDet 3.x
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='.',
        ann_file=None,  # not used by init_detector
        data_prefix=dict(img='.'),
        filter_cfg=None,
        pipeline=test_pipeline,
        metainfo=dict(classes=('person',), palette=[(220, 20, 60)]),
        test_mode=True,
    ),
)

# Optional evaluator stub (not used in init_detector)
test_evaluator = dict(type='CocoMetric', ann_file=None)
