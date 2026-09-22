# Módulo 3 adaptado ao domínio veicular, que é a adaptação que ele nunca teve.
#
# O Módulo 2 recebeu duas etapas de adaptação de domínio; o lifting é treinado
# no H3WB, com pessoas em pé num laboratório, e aplicado a um ocupante sentado.
# Aqui ele passa a ver o domínio de aplicação, com duas particularidades que o
# projeto já pagou caro para aprender:
#
# **Ensaio obrigatório.** A referência do Drive&Act cobre 23 keypoints; face e
# mãos não têm alvo e entrariam com peso zero. Treinar só nela repetiria o
# esquecimento da Etapa 3, que derrubou o whole-body AP de 0,6931 para 0,2330.
# Por isso o lote intercala H3WB, onde a supervisão é completa.
#
# **Sem corrupção simulada sobre o Drive&Act.** A entrada dele já vem corrompida
# pelo estimador real --- pernas na borda, quadril extrapolado --- e o
# `SimulatedEstimatorNoise` a ignora pela marca `corrupcao_real`. No H3WB a
# simulação continua, porque ali a entrada é ground truth.
_base_ = ['./lift3d_dstformer_h3wb_robusto_v3.py']

custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.pose_lifter',
        'mmpose.models.backbones.dstformer',
        'mmpose.models.heads.regression_heads.motion_regression_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.data.h3wb_dataset',
        'src.data.driveact_lift_dataset',
        'src.data.estimator_noise',
        'src.evaluation.wholebody_mpjpe',
    ],
    allow_failed_imports=True)

# Proporção do Drive&Act no lote. Ele tem cerca de 11 mil janelas contra 8,3 mil
# do H3WB oversampleado; com 0,5 as duas ficam próximas, e a supervisão completa
# do H3WB cobre metade dos exemplos.
DRIVEACT_SAMPLE_RATIO = 0.5

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody.py'),
        datasets=[
            dict(type='DriveActLiftDataset',
                 ann_file='data/processed/driveact/lift_train.npz',
                 seq_len=16, window_stride=8),
            dict(type='H3WBSeq2SeqDataset',
                 ann_file='h3wb_annotations.npz',
                 data_root='data/processed/h3wb/',
                 data_prefix=dict(img='original/'),
                 seq_len=16, multiple_target=16, window_stride=8,
                 keypoint_2d_src='gt',
                 oversample_actions=dict(Sitting=3, SittingDown=3)),
        ],
        sample_ratio_factor=[DRIVEACT_SAMPLE_RATIO, 1.0],
        pipeline={{_base_.train_pipeline}},
        test_mode=False))

# A validação continua no H3WB: é ela que denuncia esquecimento, que é o risco
# desta etapa. O domínio veicular se mede depois, com a régua própria.
load_from = 'work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth'
resume = False

max_epochs = 5
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

work_dir = 'work_dirs/lift3d_veicular'
