"""Catálogo dos detectores de pessoas do estágio 1.

Camada Model. São só dados: nomes, checkpoints e qual é o padrão.

Existe separado de `pose_pipeline` por um motivo medido: aquele módulo importa
MMPose e PyTorch na carga, e ler três strings de lá custa 3,2 segundos mais os
avisos do MMCV. Os controllers precisam destes nomes para montar `--detector`,
isto é, **antes** de decidir se vão carregar modelo algum --- um `--help` não
deve pagar o preço de um import de framework.
"""

from __future__ import annotations

RTMDET_CONFIG = 'configs/detectors/rtmdet_nano_person_infer.py'
RTMDET_CHECKPOINT = ('checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-'
                     '05d8511e.pth')
YOLO_CHECKPOINT = 'checkpoints/yolo26n-pose.pt'

DETECTOR_YOLO = 'yolo26n-pose'
DETECTOR_RTMDET = 'rtmdet-nano'
DETECTOR_NONE = 'nenhum'

# O padrão é o YOLO26n-pose **por medição**, não por ser mais novo: na QP1, sobre
# os mesmos 1454 quadros do Drive&Act, ele custa 21,30ms contra 33,26 do
# RTMDet-nano, com erro corporal normalizado de 0,0292 contra 0,0300 e nenhum
# quadro sem detecção. Domina nos três eixos ao mesmo tempo, e os 12ms
# recuperados são o que tira o requisito de tempo real da folga de 1% em que
# passava.
#
# A QP1 mede só o corpo, porque o Drive&Act não anota face nem mãos. O efeito da
# troca sobre essas regiões foi medido em separado, com
# `scripts/measure_region_error.py --detector`, justamente para não repetir a
# Etapa 3 --- que melhorou o corpo e desfez a face sem que nada medisse a face.
DEFAULT_DETECTOR = DETECTOR_YOLO

# Os controllers oferecem esta escolha em `--detector`; a lista vive aqui para
# não ser repetida em cada um deles.
DETECTOR_CHOICES = (DETECTOR_YOLO, DETECTOR_RTMDET, DETECTOR_NONE)
