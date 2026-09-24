# Avaliação no Drive&Act com as duas métricas lado a lado.
#
# O AP do protocolo COCO satura aqui e não distingue modelos: sem treino algum
# no domínio o RTMW-x mede 0,9351, e adaptado mede 0,9330. Ele é mantido por
# comparabilidade com a literatura, mas o critério de aceite passa a ser o erro
# normalizado por tronco, que não é inflado pela área do objeto nem pela caixa
# derivada dos próprios keypoints. Ver src/evaluation/normalized_keypoint_error.py.
_base_ = ['./rtmw_x_wholebody_eval.py']

custom_imports = dict(
    imports=[
        'mmpose.models.detectors.topdown_pose_estimator',
        'mmpose.models.backbones.cspnext',
        'mmpose.models.necks.cspnext_pafpn',
        'mmpose.models.heads.coord_cls_heads.rtmw_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.evaluation.normalized_keypoint_error',
    ],
    # topdown_pose_estimator não existe neste MMPose; a lista é herdada do config
    # base e a tolerância é o que a mantém funcional.
    allow_failed_imports=True)

data_root = 'data/processed/driveact/'
ann_file = 'driveact_midlevel.chunks_90.split_0.val.json'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='val/'),
    ))
val_dataloader = test_dataloader

test_evaluator = [
    dict(type='CocoWholeBodyMetric', ann_file=data_root + ann_file),
    dict(type='TorsoNormalizedError'),
]
val_evaluator = test_evaluator
