# Config de avaliação isolada do RTMW-x WholeBody (384x288).
#
# Espelha `rtmpose_m_wholebody_eval.py`, permitindo medir o mesmo checkpoint em
# RGB e em grayscale trocando apenas `data_root`. A arquitetura reproduz o
# config oficial `rtmw-x_8xb320-270e_cocktail14-384x288.py`; qualquer divergência
# aqui faria os pesos carregarem parcialmente e produziria métricas silenciosamente
# erradas, então os valores não devem ser ajustados sem conferir a fonte.
#
# Diferenças estruturais relevantes em relação ao RTMPose-m:
#   - backbone maior (deepen 1.33 / widen 1.25 contra 0.67 / 0.75)
#   - possui neck CSPNeXtPAFPN, ausente no RTMPose-m
#   - head RTMWHead com in_channels=1280
#   - entrada 384x288 em vez de 256x192, e sigma do codec proporcionalmente maior

default_scope = 'mmpose'

custom_imports = dict(
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.necks.cspnext_pafpn',
        'mmpose.models.heads.coord_cls_heads.rtmw_head',
        'mmpose.datasets',
        'mmpose.evaluation',
    ],
    allow_failed_imports=True)

# Trocáveis por --cfg-options
data_root = 'data/processed/grayscale/'
img_prefix = 'val2017/'
ann_file = 'annotations/coco_wholebody_val_v1.0.json'

num_keypoints = 133
input_size = (288, 384)  # (w, h)

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    decode_visibility=True)

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
        deepen_factor=1.33,
        widen_factor=1.25,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=None,
        out_indices=(1, 2),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMWHead',
        in_channels=1280,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
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
            beta=1.,
            label_softmax=True,
            label_beta=10.,
            mask=list(range(23, 91)),
            mask_weight=0.5),
        decoder=codec),
    test_cfg=dict(flip_test=True))

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

test_dataloader = dict(
    batch_size=8,  # 384x288 num modelo x: cabe em 8 GB com folga menor
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

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
visualizer = dict(type='PoseLocalVisualizer', vis_backends=[])
work_dir = 'work_dirs/eval_zero_shot'
