# Controle da QP4: o lifting de quadro único com o lote do treino base.
#
# O quadro único (lift3d_dstformer_h3wb_1frm.py, lote 64) mediu 34,75mm no S7
# contra 38,96mm da janela de 16 quadros (lote 4). O lote é a diferença de
# condição que mais poderia explicar isso, e este treino a remove: mesmo lote de
# 4, mesma taxa, mesmas 30 épocas. Restam o passo (todo quadro vira amostra) e a
# ausência do termo de velocidade, inerentes a um quadro só.
_base_ = ['./lift3d_dstformer_h3wb_1frm.py']

train_batch_size = 4
train_dataloader = dict(batch_size=train_batch_size)

work_dir = 'work_dirs/lift3d_dstformer_h3wb_1frm_lote4'
