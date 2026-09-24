#!/usr/bin/env python
"""Compara as três configurações de detecção 2D da QP1 no domínio veicular.

A pergunta é qual delas oferece a melhor relação entre acurácia e latência em
imagens infravermelhas do Drive&Act. As configurações são:

    caixa de ground truth  limite superior de acurácia; não é operável, porque
                           em operação não existe anotação
    RTMDet-nano            1,01M parâmetros
    sem detector           o frame inteiro como caixa única, viável porque a
                           câmera é rigidamente montada no habitáculo
    YOLOv11n-pose          2,87M parâmetros, keypoints descartados
    YOLOv12n               2,60M; só detecção, porque a v12 não tem pesos de pose
    YOLOv26n-pose          3,68M parâmetros

A métrica é o erro normalizado por comprimento de tronco, e **não** o AP: no
Drive&Act o AP satura a ponto de inverter qual modelo é melhor. Ver
`src/evaluation/normalized_keypoint_error.py`.

O valor aqui é a média **por quadro**, enquanto a métrica usada na validação do
treino agrega todos os keypoints de todos os quadros de uma vez. As duas são
defensáveis e não são intercambiáveis: esta pesa cada quadro igualmente, aquela
pesa cada keypoint. Não comparar os dois números diretamente.

Exemplo:
    python scripts/compare_detectors.py --max-frames 600
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# O Projeto Físico especificava YOLOv11-pose, e o levantamento de otimização
# sugeria YOLOv12. **A Ultralytics nunca publicou pesos de pose para a v12** —
# só de detecção — o que para um pipeline top-down é irrelevante: o segundo
# estágio recebe apenas o recorte da caixa e descarta keypoints produzidos antes.
# Por isso a v12 entra na comparação como detector puro, que é o papel que
# qualquer uma delas de fato cumpre aqui.
YOLO_CHECKPOINTS = {
    'YOLOv11n-pose': 'checkpoints/yolo11n-pose.pt',
    'YOLOv12n': 'checkpoints/yolo12n.pt',
    'YOLOv26n-pose': 'checkpoints/yolo26n-pose.pt',
}

# Sem detecção, o erro é indefinido. Contabilizar como zero premiaria a falha;
# contabilizar como infinito tornaria a média inútil. A taxa de quadros sem
# detecção é reportada à parte, que é a informação honesta.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=Path,
                   default=Path('data/processed/driveact/val'))
    p.add_argument('--annotations', type=Path,
                   default=Path('data/processed/driveact/'
                                'driveact_midlevel.chunks_90.split_0.val.json'))
    p.add_argument('--pose-ckpt', default=None,
                   help='Estimador 2D; o padrão é o modelo corrente do painel')
    p.add_argument('--max-frames', type=int, default=1500)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, default=Path('results/qp1_detectores.json'))
    return p.parse_args()


def main():
    args = parse_args()

    import cv2
    import torch

    from src.data.driveact import annotated_pairs
    from src.evaluation.normalized_keypoint_error import instance_error
    from src.models import torch_compat  # noqa: F401
    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline, PersonDetector,
                                          RTMDET_CHECKPOINT, RTMDET_CONFIG)
    from src.models.yolo_detector import YoloPersonDetector

    from src.models import operating_config as panel

    pairs = annotated_pairs(json.loads(args.annotations.read_text()),
                            args.max_frames)

    work_dir = Path('work_dirs/qp1')
    work_dir.mkdir(parents=True, exist_ok=True)
    pose_config = panel.config_without_flip_test(panel.POSE_CONFIG, work_dir)
    # O estimador da montagem de retrovisor, que é a montagem do Drive&Act. O
    # padrão anterior era o da mesa, e medir o domínio veicular com o modelo do
    # outro domínio mede a troca de modelo junto com o detector.
    pose_checkpoint = (args.pose_ckpt
                       or panel.POSE_CHECKPOINT_BY_MOUNTING['retrovisor'])

    configurations = {
        'caixa de ground truth': 'gt',
        'RTMDet-nano': PersonDetector(RTMDET_CONFIG, RTMDET_CHECKPOINT,
                                      args.device, DEFAULT_DETECTOR_SCORE),
        'sem detector': None,
    }
    for label, checkpoint in YOLO_CHECKPOINTS.items():
        if Path(checkpoint).exists():
            configurations[label] = YoloPersonDetector(
                checkpoint, args.device, DEFAULT_DETECTOR_SCORE)

    report = {}
    for label, detector in configurations.items():
        pipeline = FullBodyPosePipeline(
            pose_config, pose_checkpoint, args.device,
            detector=None if detector == 'gt' else detector)

        errors, latencies, missed, extra_boxes = [], [], 0, 0
        for image, annotation in pairs:
            frame = cv2.imread(str(args.frames / image['file_name']))
            if frame is None:
                continue

            started = time.perf_counter()
            if detector == 'gt':
                x, y, w, h = annotation['bbox']
                result = pipeline(frame, boxes=[[x, y, x + w, y + h]])
            else:
                result = pipeline(frame)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1e3)

            if result.num_people == 0:
                missed += 1
                continue
            # O Drive&Act tem um ocupante por quadro, então toda caixa além da
            # primeira é espúria. Ela custa duas vezes: uma passada inteira da
            # pose, e o risco de a primeira caixa não ser a do ocupante.
            if result.num_people > 1:
                extra_boxes += 1

            # O Drive&Act anota apenas corpo e pés, e o campo `keypoints` do
            # formato COCO-WholeBody carrega só as 17 juntas corporais; face e
            # mãos vivem em campos separados e aqui estão vazias.
            reference = np.array(annotation['keypoints']).reshape(-1, 3)

            # Com mais de uma detecção, a primeira é a do ocupante: o Drive&Act
            # tem uma pessoa por quadro.
            measured = instance_error(result.keypoints[0], reference)
            if measured is None:
                continue
            errors.append(measured[0])

        report[label] = {
            'erro_normalizado': round(float(np.mean(errors)), 4),
            'latencia_mediana_ms': round(float(np.median(latencies)), 2),
            'fps': round(1000.0 / float(np.median(latencies)), 1),
            'quadros_sem_deteccao': missed,
            'quadros_com_caixas_extras': extra_boxes,
            'quadros_medidos': len(errors),
        }
        print(f'  {label:24s} erro {report[label]["erro_normalizado"]:.4f}  '
              f'{report[label]["latencia_mediana_ms"]:6.2f} ms  '
              f'({report[label]["fps"]:5.1f} FPS)  '
              f'sem detecção: {missed}  caixas extras: {extra_boxes}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\nresultado em {args.out}')


if __name__ == '__main__':
    main()
