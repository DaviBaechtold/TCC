# Módulo 3 — lifting 2D para 3D full-body, 133 keypoints, janela de 16 frames.
#
# Backbone DSTFormer, que implementa exatamente a atenção espaço-temporal
# fatorizada especificada no Projeto Físico: atenção espacial entre as juntas
# dentro de cada frame, alternada com atenção temporal entre frames para cada
# junta. É essa fatorização que torna viável operar sobre 133 keypoints; a
# formulação achatada custaria 27 GFLOPs por frame.
#
# Treinado no H3WB (Human3.6M 3D WholeBody), único dataset com ground truth
# tridimensional para corpo, face e mãos. Anotações preparadas por
# `scripts/convert_h3wb.py`.
#
# Referência de calibração — benchmark H3WB tarefa 1, MPJPE em mm:
#   SimpleBaseline 125,4 | Large SimpleBaseline 112,3 | Jointformer (SOTA) 88,3
# Meta do projeto: inferior a 110mm de MPJPE full-body.

default_scope = 'mmpose'

custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.pose_lifter',
        'mmpose.models.backbones.dstformer',
        'mmpose.models.heads.regression_heads.motion_regression_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.data.h3wb_dataset',
    ],
    allow_failed_imports=False)

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
num_keypoints = 133
sequence_length = 16   # T especificado no Projeto Físico
max_epochs = 30
base_lr = 2e-4
train_batch_size = 16
val_batch_size = 16

randomness = dict(seed=42)
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ----------------------------------------------------------------------------
# Codecs
# ----------------------------------------------------------------------------
# `concat_vis=True` acrescenta a confiança do detector como terceiro canal de
# entrada, o que permite ao modelo atenuar keypoints pouco confiáveis — que no
# domínio veicular são justamente os ocluídos pelo volante.
train_codec = dict(
    type='MotionBERTLabel',
    num_keypoints=num_keypoints,
    concat_vis=True,
    mode='train')

# `rootrel=True` na avaliação: o lifting monocular determina a pose relativa à
# raiz, não a translação absoluta em relação à câmera. Avaliar sem ancorar
# mediria também um erro de profundidade que o método não se propõe a resolver.
val_codec = dict(
    type='MotionBERTLabel',
    num_keypoints=num_keypoints,
    concat_vis=True,
    rootrel=True)

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='DSTFormer',
        in_channels=3,          # (x, y, confiança)
        feat_size=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        num_keypoints=num_keypoints,
        seq_len=sequence_length,
        att_fuse=True,
    ),
    head=dict(
        type='MotionRegressionHead',
        in_channels=512,
        out_channels=3,
        embedding_size=512,
        # A perda combina erro posicional e erro de velocidade entre frames
        # consecutivos. O segundo termo penaliza oscilação temporal, que é o
        # jitter investigado pela QP4; sem ele a rede minimiza o erro de cada
        # frame isoladamente e produz sequências visualmente instáveis.
        loss=dict(type='MPJPEVelocityJointLoss'),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True))

# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
dataset_type = 'H3WBSeq2SeqDataset'
data_root = 'data/processed/h3wb/'

train_pipeline = [
    dict(type='GenerateTarget', encoder=train_codec),
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(center_mode='static', center_x=0.),
        target_flip_cfg=dict(center_mode='static', center_x=0.),
        flip_label=True),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'factor', 'camera_param')),
]

val_pipeline = [
    dict(type='GenerateTarget', encoder=val_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'factor', 'camera_param')),
]

# `multiple_target` faz a rede prever os 16 frames da janela, e não apenas um.
# Além de fornecer mais sinal de supervisão, permite medir na avaliação tanto a
# variante centrada quanto a causal, que é o trade-off entre latência e
# acurácia levantado pela QP4: a 30 FPS, esperar pelo frame central custa oito
# frames de atraso, ou 267ms, acima do orçamento de latência do projeto.
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='h3wb_annotations.npz',
        data_root=data_root,
        data_prefix=dict(img='original/'),
        # A janela e o número de alvos coincidem: a rede recebe 16 frames e
        # prevê os 3D de todos eles. Ver `src/data/h3wb_dataset.py` para a
        # guarda da classe base que precisa ser contornada.
        seq_len=sequence_length,
        multiple_target=sequence_length,
        keypoint_2d_src='gt',
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='h3wb_annotations.npz',
        data_root=data_root,
        data_prefix=dict(img='original/'),
        seq_len=sequence_length,
        multiple_target=sequence_length,
        keypoint_2d_src='gt',
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# MPJPE sem alinhamento e com alinhamento de Procrustes. O segundo remove
# ambiguidades de escala e rotação, isolando o erro de forma da pose.
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
    dict(type='MPJPE', mode='p-mpjpe'),
]
test_evaluator = val_evaluator

# ----------------------------------------------------------------------------
# Otimização
# ----------------------------------------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=200),
    dict(type='ExponentialLR', gamma=0.96, begin=1, end=max_epochs,
         by_epoch=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = []

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

visualizer = dict(
    type='Pose3dLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='TensorboardVisBackend')],
    name='visualizer')

log_level = 'INFO'
work_dir = 'work_dirs/lift3d_dstformer_h3wb'
