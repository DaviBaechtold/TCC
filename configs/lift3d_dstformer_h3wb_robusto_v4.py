# Módulo 3, quarta versão: a colocação da junta cortada passa a ser a medida.
#
# O v3 corrigiu o quadril e não transferiu para as pernas, e a medição explica
# por quê: ele prende **tudo** na linha de corte, enquanto o estimador real só
# faz isso com a junta imediatamente abaixo dela. Medido em
# `scripts/measure_absent_placement.py`, sobre 290 quadros da webcam de mesa e
# 300 do Drive&Act (`results/posicao_ausentes_*.json`), em larguras de ombro:
#
#   quadril    junto da borda em 98 e 99% dos quadros, resposta 6,1
#   joelho     43 a 53%, e 17 a 26% caem sobre o corpo visível
#   tornozelo  31 a 59%, e 17 a 44% sobre o corpo
#   pés        6 a 40%, com o percentil 10 acima da linha dos ombros
#
# `cut_placement='medido'` implementa essa mistura de três modos --- borda,
# sobre o corpo visível, espalhado --- com a probabilidade de encostar na borda
# decaindo com a distância à linha.
#
# Parte do mesmo checkpoint do v3, o v2, para que a comparação entre os dois
# isole uma variável só: o modelo de colocação.
_base_ = ['./lift3d_dstformer_h3wb_robusto_v3.py']

train_pipeline = [
    dict(type='GenerateTarget', encoder={{_base_.train_codec}}),
    dict(type='SimulatedEstimatorNoise', prob=0.6, max_groups=2,
         frame_cut_prob=0.5, unobserved_confidence=0.3,
         cut_placement='medido'),
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

work_dir = 'work_dirs/lift3d_robusto_v4'
