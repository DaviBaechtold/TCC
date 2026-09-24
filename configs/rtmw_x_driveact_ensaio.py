# Etapa 3, terceira versão: adapta o corpo ao infravermelho sem esquecer face e
# mãos.
#
# Medição que motiva este config: o checkpoint da v2 mede 0,2330 de whole-body
# AP no COCO em cinza, contra 0,6931 do modelo da Etapa 2 de onde partiu. A
# decomposição por região mostra a face errando seis vezes mais (0,0359 para
# 0,2198, normalizado pela distância interocular) e as mãos duas vezes e meia.
# A causa é a mesma da Subseção sobre supervisão parcial: no Drive&Act não há
# anotação de face nem de mãos, elas têm peso zero na perda, e nada as segura.
#
# A correção é ensaio: metade dos exemplos vem do Drive&Act, que ensina o corpo
# no domínio de aplicação, e uma parcela vem do COCO em cinza, onde face e mãos
# têm anotação e continuam recebendo gradiente. É a correção padrão para
# esquecimento catastrófico, e aqui ela não custa dado novo --- os dois
# conjuntos já estão convertidos.
#
# Parte do checkpoint da Etapa 2, e não do da v2: o que se quer é refazer a
# adaptação com a mistura certa, não remendar pesos que já esqueceram. Assim a
# comparação com a v2 isola uma variável só, que é a composição do lote.
_base_ = ['./rtmw_x_driveact_ft_v2.py']

split = 'midlevel.chunks_90.split_0'

# Proporção do subconjunto do COCO por época. O subconjunto já é filtrado pelas
# instâncias que anotam face ou mãos (`scripts/filter_wholebody_annotations.py`,
# 77.214 de 262.465): no conjunto integral só 29,4% carregam anotação ali, e
# misturar o resto diluiria por três o gradiente que o ensaio existe para
# manter. Com 0,5 ele contribui cerca de 39 mil instâncias contra as 92 mil do
# Drive&Act, ou seja, aproximadamente um exemplo em cada três supervisiona face
# ou mãos --- sem diluir a adaptação ao domínio, que é o objetivo da etapa.
COCO_SAMPLE_RATIO = 0.5

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody.py'),
        datasets=[
            dict(
                type='CocoWholeBodyDataset',
                data_root='data/processed/driveact/',
                data_mode='topdown',
                ann_file=f'driveact_{split}.train.json',
                data_prefix=dict(img='train/'),
                pipeline=[]),
            dict(
                type='CocoWholeBodyDataset',
                data_root='data/processed/grayscale/',
                data_mode='topdown',
                ann_file='annotations/coco_wholebody_train_v1.0_qualquer.json',
                data_prefix=dict(img='train2017/'),
                pipeline=[]),
        ],
        sample_ratio_factor=[1.0, COCO_SAMPLE_RATIO],
        pipeline={{_base_.train_pipeline}},
        test_mode=False))

# A validação continua sendo só a do Drive&Act, que é o que seleciona o
# checkpoint pelo erro corporal no domínio de aplicação. O que este treino
# precisa provar sobre face e mãos se mede no COCO ao final, com o instrumento
# por região --- validar nos dois a cada época dobraria o custo para responder
# uma pergunta que só interessa no fim.

load_from = ('work_dirs/rtmw_x_gray_lora/'
             'best_coco-wholebody_AP_epoch_5_merged.pth')
resume = False

# Quatro épocas: na Etapa 3 original a primeira respondeu por 96% do ganho
# corporal, e o lote aqui é um terço maior.
max_epochs = 4
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

work_dir = 'work_dirs/rtmw_x_driveact_ensaio'
