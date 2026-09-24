# Etapa 3 — especialização no domínio veicular real (Drive&Act, vista de
# retrovisor interno).
#
# Parte do checkpoint já adaptado ao grayscale sintético pela Etapa 2, com os
# adaptadores fundidos. A Etapa 2 levou o whole-body AP de 68,57% para 69,30%
# em COCO-WholeBody convertido, e saturou ali. O que sobra do gap não está no
# COCO convertido, está aqui: medido sobre 300 imagens de cada conjunto, o
# grayscale sintético tem intensidade média 105,3 e desvio 56,4, enquanto o NIR
# real do Drive&Act tem média 29,6 e desvio 31,0 — três vezes e meia mais
# escuro, com metade do contraste. Nenhuma quantidade de treino sobre COCO
# convertido cobre essa diferença.
#
# Só corpo e pés são anotados aqui, e da posição de retrovisor apenas 12 dos 17
# keypoints corporais são observáveis. Face e mãos ficam com visibilidade 0, o
# que já as exclui da perda: o codec SimCC deriva o peso de cada keypoint da
# própria visibilidade. Não é preciso mascarar por configuração, e tentar fazê-lo
# duplicaria a regra em dois lugares.
_base_ = ['./rtmw_x_wholebody_gray_lora.py']

max_epochs = 10
base_lr = 5e-5   # metade da Etapa 2: o modelo já está adaptado, aqui se especializa

data_root = 'data/processed/driveact/'
split = 'midlevel.chunks_90.split_0'

# Checkpoint da Etapa 2 com os adaptadores fundidos. Fundido, e não bruto,
# porque a injeção desta etapa espera nomes de camada comuns — injetar sobre
# nomes já adaptados criaria `conv.base.base` e o carregamento falharia em
# silêncio, que é a armadilha registrada em scripts/train_wholebody.py.
load_from = ('work_dirs/rtmw_x_gray_lora/'
             'best_coco-wholebody_AP_epoch_5_merged.pth')
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=f'driveact_{split}.train.json',
        data_prefix=dict(img='train/'),
    ))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=f'driveact_{split}.val.json',
        data_prefix=dict(img='val/'),
    ))

test_dataloader = val_dataloader

custom_imports = dict(
    imports=[
        'mmpose.models.backbones.cspnext',
        'mmpose.models.necks.cspnext_pafpn',
        'mmpose.models.heads.coord_cls_heads.rtmw_head',
        'mmpose.datasets',
        'mmpose.evaluation',
        'src.evaluation.normalized_keypoint_error',
    ],
    allow_failed_imports=True)

# O AP é mantido por comparabilidade com a literatura, mas **não** serve de
# critério aqui: ele satura neste dataset e chega a inverter qual modelo é
# melhor. Medido sobre estas mesmas 20.288 instâncias, o modelo sem adaptação
# alguma marca 0,9351 e o adaptado 0,9330, enquanto o erro em pixels vai de
# 16,17 para 15,04 — as duas métricas discordam quanto ao sinal. Ver
# src/evaluation/normalized_keypoint_error.py para as três causas da saturação.
val_evaluator = [
    dict(type='CocoWholeBodyMetric',
         ann_file=data_root + f'driveact_{split}.val.json'),
    dict(type='TorsoNormalizedError'),
]
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

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

work_dir = 'work_dirs/rtmw_x_driveact_ft'

# Selecionar o melhor checkpoint pelo AP escolheria pelo ruído da métrica
# saturada. O erro médio em pixels é o critério, e menor é melhor.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='torso/px_mean',
        rule='less',
        max_keep_ckpts=2))
