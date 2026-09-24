# Etapa 3, segunda versão: supervisiona o que a câmera não enxerga.
#
# A primeira versão funcionou no que se propunha — erro corporal de 15,04 para
# 9,52 pixels — e produziu um efeito colateral que só apareceu ao inspecionar o
# painel. As juntas fora de quadro têm peso zero na perda, nada pune uma
# predição confiante e errada ali, e a resposta nelas subiu de 3,59 para 5,13,
# com a fração acima do limiar indo de 51% para 92%. O módulo de lifting, que
# consome esses keypoints, piorou de 82,88mm para 98,33mm.
#
# Aqui elas recebem peso não nulo com alvo uniforme, que é o sinal "não sei".
# Ver src/data/absent_keypoint_supervision.py.
_base_ = ['./rtmw_x_driveact_ft.py']

custom_imports = dict(
    imports=[
        'mmpose.models.backbones.cspnext',
        'mmpose.models.necks.cspnext_pafpn',
        'mmpose.models.heads.coord_cls_heads.rtmw_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.evaluation.normalized_keypoint_error',
        'src.data.absent_keypoint_supervision',
    ],
    allow_failed_imports=True)

# A supervisão vem depois do `GenerateTarget`, que é quem calcula os pesos, e
# depois da multiplicação pelos pesos do dataset que ele aplica internamente.
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.85, 1.15], rotate_factor=20),
    dict(type='TopdownAffine', input_size={{_base_.codec.input_size}}),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            dict(type='RandomGamma', gamma_limit=(70, 130), p=0.4),
            dict(type='GaussNoise', var_limit=(10.0, 60.0), p=0.3),
            dict(type='Blur', blur_limit=3, p=0.1),
            dict(type='CoarseDropout',
                 max_holes=1, max_height=0.3, max_width=0.3,
                 min_holes=1, min_height=0.1, min_width=0.1, p=0.4),
        ]),
    dict(type='GenerateTarget', encoder={{_base_.codec}},
         use_dataset_keypoint_weights=True),
    dict(type='SuperviseAbsentKeypoints', weight=0.3),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# Parte do checkpoint da primeira versão, e não do da Etapa 2: o que se corrige
# é um efeito colateral, não o treino inteiro.
load_from = ('work_dirs/rtmw_x_driveact_ft/'
             'best_torso_px_mean_epoch_10_merged.pth')
resume = False

max_epochs = 6
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

work_dir = 'work_dirs/rtmw_x_driveact_ft_v2'
