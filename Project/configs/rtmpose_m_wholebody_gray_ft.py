# RTMPose-m WholeBody (256x192) — fine-tuning para o domínio grayscale/IR.
#
# Diferenças em relação ao config `rtmpose_m_wholebody_minimal.py` que produziu
# o AP=0.4373 (e que era, na prática, um smoke test):
#
#  1. `load_from` aponta para o checkpoint OFICIAL de wholebody (133 keypoints),
#     não para o de body7 (17 keypoints). Com body7 a head de 133 saídas nascia
#     aleatória e precisava reaprender face e mãos do zero. Partindo do
#     checkpoint correto o ponto de partida é AP=0.5255 em grayscale
#     (medido, ver work_dirs/eval_zero_shot/official_gray.json).
#
#  2. Pipeline de augmentation completo. O config antigo tinha apenas
#     RandomFlip, o que produzia acc_pose~0.93 no treino contra AP=0.44 na
#     validação — overfitting clássico.
#
#  3. Augmentation fotométrica escolhida para o domínio IR em vez do
#     `PhotometricDistortion` padrão. Em imagens grayscale (R=G=B) os termos de
#     matiz e saturação do PhotometricDistortion são no-op, então ele degenera
#     em brilho/contraste. Aqui isso é explicitado e complementado com gama e
#     ruído gaussiano, cobrindo as QP3 (variação de iluminação) e o ruído de
#     sensor IR descrito na definição do problema.
#
#  4. Treino em dois estágios com EMA, como na receita oficial do RTMPose:
#     augmentation agressiva no início, atenuada nas últimas épocas.

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

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
max_epochs = 60
stage2_epochs = 10  # últimas épocas com augmentation atenuada
base_lr = 5e-4      # fine-tuning: ~1/8 do LR de treino do zero
train_batch_size = 64
val_batch_size = 32

randomness = dict(seed=42)

load_from = 'checkpoints/rtmpose-m_wholebody_official_256x192.pth'
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/processed/grayscale/'

input_size = (192, 256)  # (w, h)

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------

# Estágio 1: augmentation geométrica + fotométrica agressiva.
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=input_size),
    dict(
        type='Albumentation',
        transforms=[
            # Variação de iluminação — QP3 do documento.
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            dict(type='RandomGamma', gamma_limit=(70, 130), p=0.3),
            # Ruído de sensor IR e desfoque por movimento.
            dict(type='GaussNoise', var_limit=(10.0, 60.0), p=0.3),
            dict(type='Blur', blur_limit=3, p=0.1),
            dict(type='MedianBlur', blur_limit=3, p=0.1),
            # Oclusão sintética — proxy para volante, painel, cinto.
            dict(type='CoarseDropout',
                 max_holes=1, max_height=0.4, max_width=0.4,
                 min_holes=1, min_height=0.2, min_width=0.2, p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# Estágio 2: augmentation atenuada nas últimas épocas, para o modelo assentar
# na distribuição real de validação.
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform',
         shift_factor=0., scale_factor=[0.75, 1.25], rotate_factor=60),
    dict(type='TopdownAffine', input_size=input_size),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            dict(type='Blur', blur_limit=3, p=0.05),
            dict(type='MedianBlur', blur_limit=3, p=0.05),
            dict(type='CoarseDropout',
                 max_holes=1, max_height=0.2, max_width=0.2,
                 min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]
test_pipeline = val_pipeline

# ----------------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
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
    batch_size=val_batch_size,
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
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator

# ----------------------------------------------------------------------------
# Otimização
# ----------------------------------------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.02,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs - max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ----------------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco-wholebody/AP',
        rule='greater',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_epochs,
        switch_pipeline=train_pipeline_stage2),
]

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
work_dir = 'work_dirs/rtmpose_m_gray_ft'
