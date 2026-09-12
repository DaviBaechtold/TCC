#!/usr/bin/env python
"""Valida o lifting 3D no domínio veicular, contra a referência do Drive&Act.

O Módulo 2 tem duas etapas de adaptação de domínio; o Módulo 3 não tem nenhuma:
ele é treinado no H3WB, com pessoas em pé num laboratório, e aplicado a um
ocupante sentado num habitáculo. Esta é a primeira medição desse salto.

Reporta **PA-MPJPE**, com alinhamento de Procrustes, e não MPJPE. O motivo não é
preferência: sem calibração da câmera o fator de escala do lifting é
desconhecido, e o MPJPE absoluto mediria sobretudo esse fator, não a pose. O
alinhamento remove escala e rotação e isola o que se quer avaliar, que é a forma.

Duas ressalvas que acompanham qualquer número daqui:

1. A referência do Drive&Act vem de triangulação por OpenPose, e não de captura
   com marcadores. Ela própria é estimativa, o que torna esta uma análise
   exploratória e não critério de aceite.
2. Só os keypoints observáveis da posição de retrovisor entram na conta. Os
   demais não estão na imagem, e cobrar o modelo por eles mediria outra coisa.

Exemplo:
    python scripts/validate_lifting_driveact.py \\
        --poses ~/Downloads/extracted/openpose_3d --max-sequences 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEQUENCE_LENGTH = 16

from src.models.observability import MIRROR_VIEW_OBSERVABLE as OBSERVABLE_KEYPOINTS

ROOT_KEYPOINT = 0



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=Path,
                   default=Path('data/processed/driveact/val'))
    p.add_argument('--annotations', type=Path,
                   default=Path('data/processed/driveact/'
                                'driveact_midlevel.chunks_90.split_0.val.json'))
    p.add_argument('--poses', type=Path, required=True,
                   help='Diretório openpose_3d/ com a referência tridimensional')
    p.add_argument('--max-sequences', type=int, default=5)
    p.add_argument('--max-frames-per-sequence', type=int, default=200,
                   help='Teto por sequência. Cada quadro custa a pose mais o '
                        'lifting, e uma sequência inteira do Drive&Act tem '
                        'milhares — sem teto a medição leva mais tempo que o '
                        'treino que ela deveria avaliar')
    p.add_argument('--pose-ckpt', default=None,
                   help='Estimador 2D a usar. O padrão é o modelo corrente do '
                        'painel; passar outro permite medir quanto a adaptação '
                        'do Módulo 2 melhora o 3D que depende dela')
    p.add_argument('--lift-cfg', default=None,
                   help='Config do lifting; o padrão é o do painel')
    p.add_argument('--lift-ckpt', default=None,
                   help='Checkpoint do lifting. Permite comparar o modelo base '
                        'com o treinado para tolerar entrada incompleta')
    p.add_argument('--confidence', default='raw',
                   choices=['raw', 'constante', 'normalizada'],
                   help='Escala do terceiro canal da entrada do lifting. '
                        '`raw` entrega a resposta do SimCC como vem, que vai a '
                        '10; `normalizada` divide pela resposta média nos '
                        'keypoints observados; `constante` entrega 1,0, que é '
                        'o que o treino original do H3WB viu')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path,
                   default=Path('results/lifting_driveact.json'))
    return p.parse_args()


def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Alinha `predicted` a `target` por similaridade, sem deformar a pose.

    Rotação, escala e translação são livres; a forma não. É o alinhamento do
    PA-MPJPE, e remove exatamente as três grandezas que uma câmera não calibrada
    deixa indeterminadas.
    """
    predicted_center = predicted - predicted.mean(axis=0)
    target_center = target - target.mean(axis=0)

    covariance = predicted_center.T @ target_center
    left, singular, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:          # evita reflexão
        right[-1] *= -1
        rotation = right.T @ left.T
        singular[-1] *= -1

    scale = singular.sum() / (predicted_center ** 2).sum()
    return scale * (predicted_center @ rotation.T) + target.mean(axis=0)


def main():
    import json

    args = parse_args()

    from src.models import torch_compat  # noqa: F401
    from src.data.driveact import read_pose_csv
    from src.models.pose_pipeline import FullBodyPosePipeline
    from src.models.sequence_lifter import (OBSERVED_RESPONSE,
                                            SequenceLifter)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)

    annotations = json.loads(args.annotations.read_text())
    by_sequence: dict[str, list[dict]] = {}
    for image in annotations['images']:
        by_sequence.setdefault(image['file_id'], []).append(image)

    work_dir = Path('work_dirs/lifting_driveact')
    work_dir.mkdir(parents=True, exist_ok=True)
    pose = FullBodyPosePipeline(
        panel._config_without_flip_test(panel.POSE_CONFIG, work_dir),
        args.pose_ckpt or panel.POSE_CHECKPOINT, args.device, detector=None)
    # Quem normaliza é o lifter, e só ele. Antes o script dividia pela escala e
    # o lifter dividia de novo, entregando confiança na casa de 0,1 a um modelo
    # treinado entre 0,37 e 1,0 — e o resultado dessa medição foi descartado.
    escala = OBSERVED_RESPONSE if args.confidence == 'normalizada' else 1.0
    lifter = SequenceLifter(args.lift_cfg or panel.LIFT_CONFIG,
                            args.lift_ckpt or panel.LIFT_CHECKPOINT,
                            args.device, response_scale=escala)

    import cv2

    errors: list[float] = []
    por_sequencia: dict[str, float] = {}
    for file_id in sorted(by_sequence)[:args.max_sequences]:
        subject, run = file_id.split('/')
        csv_path = args.poses / subject / f'{run}.openpose.3d.csv'
        if not csv_path.exists():
            print(f'  {file_id}: sem referência 3D, ignorado')
            continue

        reference = {frame.frame_id: frame for frame in read_pose_csv(csv_path)}
        images = sorted(by_sequence[file_id],
                        key=lambda i: i['frame_id'])[:args.max_frames_per_sequence]
        lifter.reset()
        matched = 0

        for image in images:
            frame = cv2.imread(str(args.frames / image['file_name']))
            if frame is None:
                continue
            result = pose(frame)
            if not result.num_people:
                continue

            height, width = frame.shape[:2]
            scores = (np.ones_like(result.scores[0])
                      if args.confidence == 'constante' else result.scores[0])
            predicted = lifter(result.keypoints[0], scores, (width, height))
            if lifter.warming_up or image['frame_id'] not in reference:
                continue

            truth = reference[image['frame_id']].points_3d
            visible = reference[image['frame_id']].confidence > 0
            usable = [k for k in OBSERVABLE_KEYPOINTS if visible[k]]
            if len(usable) < 6:   # Procrustes sobre poucos pontos é instável
                continue

            aligned = procrustes_align(predicted[usable], truth[usable])
            errors.append(
                np.linalg.norm(aligned - truth[usable], axis=-1).mean() * 1000)
            matched += 1

        # Por sequência, e não só agregado: a referência tem erro correlacionado
        # dentro de uma sequência — mesma pessoa, mesma calibração, mesma pose
        # de fundo — e a média global esconde se uma diferença é consistente ou
        # se veio de uma sequência só.
        if matched:
            por_sequencia[file_id] = round(
                float(np.mean(errors[-matched:])), 2)
        print(f'  {file_id}: {matched} quadros comparados'
              f'{f", PA-MPJPE {por_sequencia[file_id]:.1f}mm" if matched else ""}')

    if not errors:
        raise SystemExit('nenhum quadro comparável')

    report = {
        'pa_mpjpe_mm': round(float(np.mean(errors)), 2),
        'pa_mpjpe_median_mm': round(float(np.median(errors)), 2),
        'frames': len(errors),
        'keypoints': list(OBSERVABLE_KEYPOINTS),
        'alignment': 'procrustes',
        'reference': 'Drive&Act OpenPose 3D (triangulação, não marcadores)',
        'pose_checkpoint': Path(args.pose_ckpt or panel.POSE_CHECKPOINT).name,
        'lift_checkpoint': Path(args.lift_ckpt or panel.LIFT_CHECKPOINT).name,
        'confidence': args.confidence,
        'por_sequencia': por_sequencia,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
