# Módulo 3 treinado para operar com entrada incompleta.
#
# Medição que motiva este config: o lifting mede 42,20mm de erro corporal no
# próprio domínio de treino e 82,88mm no habitáculo, e a causa não é postura nem
# ponto de vista, e sim o que a rede recebe onde não há observação.
#
# A primeira tentativa apagava esses keypoints e **piorou** o domínio veicular,
# para 108,37mm. O erro foi de raciocínio: zerar as pernas no H3WB dava 80,75mm,
# perto dos 82,88mm reais, e disso concluiu-se que a corrupção real eram zeros.
# Coincidiram as magnitudes, não os mecanismos — o estimador 2D coloca a junta
# invisível junto da borda inferior do recorte, não na origem.
#
# Esta versão simula o que foi medido, e torna informativo o canal de confiança,
# que no H3WB é constante em 1,0 e que a rede aprendeu a ignorar. Ver
# `src/data/estimator_noise.py`. Parte do checkpoint já convergido: o que falta
# não é aprender a tarefa.
_base_ = ['./lift3d_dstformer_h3wb_16frm.py']

max_epochs = 15
base_lr = 5e-5   # metade do treino original: é ajuste, não aprendizado inicial

load_from = 'work_dirs/lift3d_dstformer_h3wb/best_MPJPE_whole_epoch_30.pth'
resume = False

custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.pose_lifter',
        'mmpose.models.backbones.dstformer',
        'mmpose.models.heads.regression_heads.motion_regression_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.data.h3wb_dataset',
        'src.data.estimator_noise',
        'src.evaluation.wholebody_mpjpe',
    ],
    allow_failed_imports=True)

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# A ordem importa: a simulação vem depois do `GenerateTarget`, que é quem produz
# `keypoint_labels`, e antes do `RandomFlipAroundRoot`, para que o espelhamento
# troque os lados de um membro já extrapolado em vez de reintroduzi-lo.
train_pipeline = [
    dict(type='GenerateTarget', encoder={{_base_.train_codec}}),
    dict(type='SimulatedEstimatorNoise', prob=0.6, max_groups=2),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=200),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        end=max_epochs,
        T_max=max_epochs - 1,
        by_epoch=True,
        convert_to_iter_based=True),
]

work_dir = 'work_dirs/lift3d_robusto_v2'
