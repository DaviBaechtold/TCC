# Lifting de quadro único, para medir o ganho da janela temporal (QP4).
#
# Mesma rede, mesma perda, mesmas épocas e mesmo agendamento do treino base de
# 16 quadros (lift3d_dstformer_h3wb_16frm.py); muda só o contexto temporal. A
# comparação que interessa é com esse treino base, e não com o v3, porque o v3
# foi ajustado depois com corrupção da entrada.
#
# Três diferenças de condição, inevitáveis e declaradas junto do resultado:
# passo 1 (todo quadro de treino vira uma amostra, porque com janela de um
# quadro não há sobreposição a evitar) e lote de 64 (com um quadro a memória
# sobra; o lote de 4 do base era o teto dos 8 GB para 16 quadros). Por época, o
# base vê cerca de 112 mil alvos (7.044 janelas x 16) e este vê 60 mil.
_base_ = ['./lift3d_dstformer_h3wb_16frm.py']

sequence_length = 1
train_batch_size = 64

# `sequence_length` do pai já foi resolvido dentro dos dicionários; cada lugar
# em que a janela aparece precisa ser sobrescrito aqui.
model = dict(
    backbone=dict(seq_len=sequence_length),
    # Com um quadro não há velocidade: na perda original o termo vira a média de
    # um tensor vazio, NaN na primeira iteração. A perda ponderada devolve zero
    # nesse caso e coincide com a original nos outros dois termos; o peso da
    # velocidade é zerado para deixar isso explícito. É a terceira diferença de
    # condição em relação ao base.
    head=dict(loss=dict(_delete_=True, type='WeightedMPJPEVelocityLoss',
                        lambda_velocity=0.0)))

custom_imports = dict(
    imports=['mmpose.models.pose_estimators.pose_lifter',
             'mmpose.models.backbones.dstformer',
             'mmpose.models.heads.regression_heads.motion_regression_head',
             'mmpose.datasets', 'mmpose.evaluation',
             'src.data.h3wb_dataset', 'src.evaluation.wholebody_mpjpe',
             'src.models.lifting_loss'],
    allow_failed_imports=False)

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(seq_len=sequence_length, multiple_target=sequence_length,
                 window_stride=1))

val_dataloader = dict(
    dataset=dict(seq_len=sequence_length, multiple_target=sequence_length,
                 window_stride=1))
test_dataloader = val_dataloader

work_dir = 'work_dirs/lift3d_dstformer_h3wb_1frm'
