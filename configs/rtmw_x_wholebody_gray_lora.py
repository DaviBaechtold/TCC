# RTMW-x WholeBody (384x288) — adaptação do domínio grayscale por LoRA.
#
# Parte do checkpoint oficial cocktail14, que mede 0.6857 de whole-body AP em
# grayscale sem treino algum (`results/baselines/rtmwx_gray.json`). Esse valor é
# o piso: qualquer resultado abaixo dele significa que o treino piorou o modelo.
#
# A receita difere deliberadamente da usada no fine-tuning do RTMPose-m, que
# fracassou (0.5255 -> 0.5137 em dez épocas). Três mudanças, cada uma atacando
# uma causa identificada daquele fracasso:
#
#  1. Os pesos originais ficam congelados e só adaptadores de posto baixo são
#     treinados, o que torna impossível repetir o catastrophic forgetting.
#  2. A taxa de aprendizado cai de 5e-4 para 1e-4.
#  3. A augmentation geométrica é atenuada. O problema aqui é de domínio
#     cromático, não de aprender pose: rotações de ±80 graus e recortes
#     agressivos apenas dificultam a tarefa sem endereçar o deslocamento
#     espectral, que é o que se quer corrigir.

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

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
max_epochs = 5
base_lr = 1e-4
train_batch_size = 12   # 384x288 com backward através do backbone: ~3,2 GB
val_batch_size = 8

randomness = dict(seed=42)
load_from = ('checkpoints/'
             'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth')
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Consumido por `scripts/train_wholebody.py`, não pelo MMEngine.
lora = dict(rank=16, include=('backbone', 'neck'), trainable=('head', ))

# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/processed/grayscale/'

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

# ----------------------------------------------------------------------------
# Modelo — precisa reproduzir exatamente o config oficial, ou os pesos carregam
# parcialmente e as métricas ficam silenciosamente erradas.
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

# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    # Amplitude reduzida em relação à receita de treino do zero: o modelo já
    # sabe estimar pose, e o que se está corrigindo é o domínio cromático.
    dict(type='RandomBBoxTransform', scale_factor=[0.85, 1.15], rotate_factor=20),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
        transforms=[
            # Estas são as transformações que de fato endereçam o domínio:
            # variação de iluminação (QP3) e ruído de sensor infravermelho.
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            dict(type='RandomGamma', gamma_limit=(70, 130), p=0.4),
            dict(type='GaussNoise', var_limit=(10.0, 60.0), p=0.3),
            dict(type='Blur', blur_limit=3, p=0.1),
            # Oclusão sintética como proxy de volante, painel e cinto, com
            # probabilidade moderada em vez de aplicada a toda amostra.
            dict(type='CoarseDropout',
                 max_holes=1, max_height=0.3, max_width=0.3,
                 min_holes=1, min_height=0.1, min_width=0.1, p=0.4),
        ]),
    dict(type='GenerateTarget', encoder=codec, use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]
test_pipeline = val_pipeline

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
# Precisão mista em bfloat16, e não em float16.
#
# Com fp16 o treino produzia NaN desde a primeira iteração, embora o forward
# fosse comprovadamente saudável (loss 0,0013 e acc_pose 0,98 num passo isolado).
# A causa é uma interação entre o transbordo de expoente do fp16 e o recorte de
# gradiente: quando a norma total vira `inf`, o coeficiente de recorte
# `max_norm / total_norm` vira `nan` e contamina os gradientes *depois* de o
# GradScaler já ter verificado transbordo, de modo que o passo não é descartado
# e os pesos são corrompidos de forma irreversível.
#
# O bfloat16 tem o mesmo alcance de expoente do float32 e não transborda, além
# de dispensar escalonamento de perda.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=300),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        end=max_epochs,
        T_max=max_epochs - 1,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco-wholebody/AP',
        rule='greater',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# EMA deliberadamente ausente: com os pesos base congelados, a média móvel
# atuaria apenas sobre adaptadores que partem de zero, atrasando a adaptação
# sem o efeito estabilizador que ela tem num treino completo.
custom_hooks = []

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='TensorboardVisBackend')],
    name='visualizer')

log_level = 'INFO'
work_dir = 'work_dirs/rtmw_x_gray_lora'
