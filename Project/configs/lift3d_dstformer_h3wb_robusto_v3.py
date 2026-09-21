# Módulo 3 treinado para operar com o corpo cortado por uma linha horizontal.
#
# Medição que motiva este config: o v2 nunca viu um quadril escondido, porque
# `KEYPOINT_GROUPS` não contém os índices 11 e 12 em grupo algum. A webcam de
# mesa entrega exatamente isso — a Etapa 2 coloca os dois quadris em y mediano
# 706 e 708 num quadro de 720 px, 13 px acima da borda, e com resposta 6,1 de
# 8,3, que passa o limiar de detecção em 95% dos quadros. O modelo recebe um
# quadril que *parece* observado e está errado.
#
# Sob esse corte aplicado ao H3WB com verdade de campo, o v2 erra 507mm nos
# quadris: o tronco colapsa e os quadris sobem à altura dos ombros. As pernas
# erram 239mm, e caem para 180mm quando os quadris verdadeiros são devolvidos —
# o erro das pernas é consequência do quadril perdido, não das pernas. Nas
# janelas sentadas, que são o caso de aplicação, as pernas ficam em 356mm.
#
# Esta versão troca o grupo anatômico pela linha de corte na metade dos
# exemplos, casa a confiança das juntas cortadas com o teto que o painel aplica
# em inferência, e repondera as ações sentadas, que o Human3.6M quase não tem.
# Ver `src/data/estimator_noise.py`. Parte do checkpoint do v2: o que falta não
# é aprender a tarefa nem a extrapolação por grupo.
_base_ = ['./lift3d_dstformer_h3wb_robusto.py']

max_epochs = 15
base_lr = 5e-5   # mesmo do v2: é ajuste sobre um modelo convergido, não treino

load_from = 'work_dirs/lift3d_robusto_v2/best_MPJPE_whole_epoch_15.pth'
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

# Metade dos exemplos recebe o corte; dos 50% restantes, 60% recebem o grupo
# anatômico do v2 e o resto chega limpo. A exposição ao grupo cai de 60% para
# 30%, e isso é aceito: um corte entre quadril e joelho já remove pernas e pés,
# que é o grupo de maior peso e o único que a vista de retrovisor perde inteiro.
#
# `unobserved_confidence` precisa ser igual a
# `src.data.estimator_noise.UNOBSERVED_CONFIDENCE_CAP`, que é o teto que o
# painel aplica às juntas que ele sabe não ter observado. Treino e inferência
# veem a mesma faixa ou o canal de confiança volta a ser ruído.
train_pipeline = [
    dict(type='GenerateTarget', encoder={{_base_.train_codec}}),
    dict(type='SimulatedEstimatorNoise', prob=0.6, max_groups=2,
         frame_cut_prob=0.5, unobserved_confidence=0.3),
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

# Fator 3 leva as janelas sentadas de 8,97% para 22,8% do conjunto de treino
# (632 janelas distintas de 7.044 passam a 1.896 de 8.308). Três, e não mais:
# as janelas distintas continuam sendo 632, de modo que um fator maior repete as
# mesmas janelas em vez de acrescentar informação, e tira peso das ações em pé,
# que são de onde vem a geometria quadril-perna em geral.
#
# `SittingDown` entra junto por ser a transição: ela cobre as posturas entre
# estar de pé e estar sentado, que nenhuma das duas ações cobre.
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        oversample_actions=dict(Sitting=3, SittingDown=3)))

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

work_dir = 'work_dirs/lift3d_robusto_v3'
