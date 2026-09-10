# RTMDet-nano para detecção de pessoas (inferência).
#
# Os hiperparâmetros aqui foram derivados do próprio checkpoint
# `rtmdet_nano_8xb32-100e_coco-obj365-person`, e não devem ser ajustados sem
# reconferir contra ele. A versão anterior deste arquivo declarava convoluções
# densas e deepen_factor=0.167, o que fazia praticamente todo o backbone, o
# neck e a head falharem ao carregar: o modelo rodava com pesos aleatórios e
# ainda assim produzia caixas, apenas sem valor algum.
#
# Evidências extraídas do checkpoint:
#   - chaves `depthwise_conv`/`pointwise_conv`            -> use_depthwise=True
#   - stem 3->8->8->16 canais                             -> widen_factor=0.25
#   - 1, 2, 2 e 1 bloco CSP nos quatro estágios           -> deepen_factor=0.33
#   - `bbox_head.rtm_cls` indexado por nível (.0/.1/.2)   -> share_conv=False

default_scope = 'mmdet'

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
        deepen_factor=0.33,
        widen_factor=0.25,
        channel_attention=True,
        use_depthwise=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        expand_ratio=0.5,
        use_depthwise=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,  # apenas a classe pessoa
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        share_conv=False,
        exp_on_reg=False,
        use_depthwise=True,
        # O checkpoint não traz pesos de `rtm_obj`. Mantê-lo ligado criaria um
        # ramo de objectness aleatório multiplicando os scores de classificação,
        # o que zera as detecções sem qualquer erro visível.
        with_objectness=False,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# 320x320 é a resolução em que este checkpoint foi treinado. A versão anterior
# usava 640x640, o que além de custar quatro vezes mais também avaliava o
# modelo fora da escala para a qual seus âncoras foram calibrados.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

# init_detector do MMDet 3.x exige um test_dataloader declarado, ainda que a
# inferência a partir de array numpy não o utilize.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='.',
        ann_file=None,
        data_prefix=dict(img='.'),
        filter_cfg=None,
        pipeline=test_pipeline,
        metainfo=dict(classes=('person', ), palette=[(220, 20, 60)]),
        test_mode=True,
    ),
)

test_evaluator = dict(type='CocoMetric', ann_file=None)
