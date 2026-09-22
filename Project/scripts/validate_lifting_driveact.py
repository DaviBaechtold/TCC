#!/usr/bin/env python
"""Valida o lifting 3D no domínio veicular, contra a referência do Drive&Act.

O Módulo 2 tem duas etapas de adaptação de domínio; o Módulo 3 não tem nenhuma:
ele é treinado no H3WB, com pessoas em pé num laboratório, e aplicado a um
ocupante sentado num habitáculo. Esta é a primeira medição desse salto.

Reporta **PA-MPJPE**, com alinhamento de Procrustes, e também MPJPE absoluto. O
alinhamento remove escala e rotação e isola a forma da pose; o absoluto mede a
escala junto, e só é interpretável porque a calibração do retrovisor é conhecida
— lente de 567px, ocupante a 0,664m da câmera, medido na própria referência 3D.
Os dois são necessários: um lifting pode acertar a forma e errar o tamanho, e é
exatamente isso que acontece quando o 2D entra fora da escala de treino.

A escala entra por dois lugares independentes, e confundi-los já descartou uma
medição. `--normalizacao camera` reprojeta o 2D na geometria em que o H3WB
treinou, corrigindo a **entrada**; o fator de decodificação corrige a **saída**.
Com a entrada 3,8 vezes fora de escala, nenhum fator de saída conserta a pose.

Duas ressalvas que acompanham qualquer número daqui:

1. A referência do Drive&Act vem de triangulação por OpenPose, e não de captura
   com marcadores. Ela própria é estimativa, o que torna esta uma análise
   exploratória e não critério de aceite.
2. Só os keypoints observáveis da posição de retrovisor entram na conta. Os
   demais não estão na imagem, e cobrar o modelo por eles mediria outra coisa.

Exemplo:
    python scripts/validate_lifting_driveact.py \\
        --poses ~/Downloads/extracted/openpose_3d --max-sequences 5

`--normalizacao largura` reproduz o caminho antigo, e é sob ele que os números
publicados antes desta correção se repetem.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEQUENCE_LENGTH = 16

from src.evaluation.pose_alignment import procrustes_align
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
    p.add_argument('--normalizacao', default='camera',
                   choices=['camera', 'largura'],
                   help='Como o 2D chega ao lifting. `camera` reprojeta os '
                        'pontos na geometria em que o H3WB treinou, usando a '
                        'calibração do retrovisor; `largura` normaliza pela '
                        'largura do quadro, que é o caminho antigo e o único '
                        'sob o qual os números já publicados se reproduzem')
    p.add_argument('--factor', type=float, default=None,
                   help='Escala de decodificação, só usada com `--normalizacao '
                        'largura`. O padrão deriva da calibração do Drive&Act '
                        '— lente de 567px, ocupante a 0,664m — em vez de usar a '
                        'mediana do H3WB, que amplia a pose em 3,8 vezes neste '
                        'domínio')
    p.add_argument('--detector', action='store_true',
                   help='Usa o detector de pessoas em vez do quadro inteiro '
                        'como caixa. É a condição de operação, e a do conjunto '
                        'de treino veicular; sem a flag reproduz as medições '
                        'anteriores desta régua')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path,
                   default=Path('results/lifting_driveact.json'))
    return p.parse_args()


def main():
    import json

    args = parse_args()

    from src.evaluation.bone_consistency import (MIN_FRAMES, consistency,
                                                 motion)
    from src.models import torch_compat  # noqa: F401
    from src.data.driveact import read_pose_csv
    from src.models.pose_pipeline import FullBodyPosePipeline
    from src.models.sequence_lifter import (DRIVEACT_FOCAL_PX,
                                            DRIVEACT_OCCUPANT_DEPTH_M,
                                            DRIVEACT_PRINCIPAL_POINT_PX,
                                            OBSERVED_RESPONSE, CameraView,
                                            SequenceLifter, factor_from_camera)

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
    # Sem detector o quadro inteiro vira caixa única, que é como todas as
    # medições anteriores desta régua foram feitas. Com `--detector` a entrada
    # passa a ser a de operação, e é sob ela que o conjunto de treino veicular
    # foi extraído --- comparar treino e medida exige a mesma condição.
    detector = None
    if args.detector:
        from src.models.pose_pipeline import DEFAULT_DETECTOR_SCORE, PersonDetector
        detector = PersonDetector(panel.DETECTOR_CONFIG,
                                  panel.DETECTOR_CHECKPOINT, args.device,
                                  DEFAULT_DETECTOR_SCORE)
    pose = FullBodyPosePipeline(
        panel._config_without_flip_test(panel.POSE_CONFIG, work_dir),
        args.pose_ckpt or panel.POSE_CHECKPOINT, args.device, detector)
    # Quem normaliza é o lifter, e só ele. Antes o script dividia pela escala e
    # o lifter dividia de novo, entregando confiança na casa de 0,1 a um modelo
    # treinado entre 0,37 e 1,0 — e o resultado dessa medição foi descartado.
    escala = OBSERVED_RESPONSE if args.confidence == 'normalizada' else 1.0
    fator = args.factor if args.factor is not None else factor_from_camera(
        DRIVEACT_FOCAL_PX, DRIVEACT_OCCUPANT_DEPTH_M)
    # Os 29 arquivos de calibração do retrovisor do Drive&Act trazem a mesma
    # câmera, e o quadro de 1280x1024 que eles declaram é o das imagens em
    # data/processed/driveact/val. Com `largura` a câmera não é construída, e
    # é a largura do quadro que normaliza — o caminho que produziu os números
    # já publicados.
    camera = (None if args.normalizacao == 'largura' else
              CameraView(DRIVEACT_FOCAL_PX, DRIVEACT_PRINCIPAL_POINT_PX,
                         DRIVEACT_OCCUPANT_DEPTH_M))
    lifter = SequenceLifter(args.lift_cfg or panel.LIFT_CONFIG,
                            args.lift_ckpt or panel.LIFT_CHECKPOINT,
                            args.device, camera=camera, factor=fator,
                            response_scale=escala)

    import cv2

    errors: list[float] = []
    absolutos: list[float] = []
    por_sequencia: dict[str, float] = {}
    coerencia: list[float] = []
    coerencia_forma: list[float] = []
    movimento: list[float] = []
    for file_id in sorted(by_sequence)[:args.max_sequences]:
        subject, run = file_id.split('/')
        csv_path = args.poses / subject / f'{run}.openpose.3d.csv'
        if not csv_path.exists():
            print(f'  {file_id}: sem referência 3D, ignorado')
            continue

        reference = {frame.frame_id: frame for frame in read_pose_csv(csv_path)}
        predicoes: list[np.ndarray] = []
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
            if lifter.warming_up:
                continue
            # A coerência de osso não precisa de referência, então acumula
            # mesmo nos quadros em que o Drive&Act não tem anotação.
            predicoes.append(predicted)
            if image['frame_id'] not in reference:
                continue

            truth = reference[image['frame_id']].points_3d
            visible = reference[image['frame_id']].confidence > 0
            usable = [k for k in OBSERVABLE_KEYPOINTS if visible[k]]
            if len(usable) < 6:   # Procrustes sobre poucos pontos é instável
                continue

            aligned = procrustes_align(predicted[usable], truth[usable])
            errors.append(
                np.linalg.norm(aligned - truth[usable], axis=-1).mean() * 1000)

            # MPJPE absoluto, só ancorado na raiz. Mede escala junto com forma,
            # e por isso é o número que denuncia entrada fora de escala — o que
            # o PA-MPJPE, que alinha a escala antes de medir, esconde.
            raiz_pred = predicted[usable] - predicted[ROOT_KEYPOINT]
            raiz_ref = truth[usable] - truth[ROOT_KEYPOINT]
            absolutos.append(
                np.linalg.norm(raiz_pred - raiz_ref, axis=-1).mean() * 1000)
            matched += 1

        # Por sequência, e não só agregado: a referência tem erro correlacionado
        # dentro de uma sequência — mesma pessoa, mesma calibração, mesma pose
        # de fundo — e a média global esconde se uma diferença é consistente ou
        # se veio de uma sequência só.
        if matched:
            por_sequencia[file_id] = round(
                float(np.mean(errors[-matched:])), 2)
        if len(predicoes) >= MIN_FRAMES:
            desvios = consistency(np.stack(predicoes))
            if 'mediana' in desvios:
                coerencia.append(desvios['mediana'])
            if 'mediana_relativa' in desvios:
                coerencia_forma.append(desvios['mediana_relativa'])
            movimento.append(motion(np.stack(predicoes)))
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
        'detector': bool(args.detector),
        'normalizacao': args.normalizacao,
        # Com `camera` quem decodifica é a geometria virtual do H3WB, e este
        # fator não é aplicado; fica no relatório só para o modo `largura`.
        'factor': round(fator, 4) if camera is None else None,
        # MPJPE absoluto, sem alinhamento: só é interpretável com o fator
        # correto, e por isso não era reportado antes.
        'mpjpe_mm': (round(float(np.mean(absolutos)), 2)
                     if absolutos else None),
        'por_sequencia': por_sequencia,
        # Coerência de osso: desvio do comprimento, que é fisicamente constante.
        # Não precisa de referência, e por isso não herda a incerteza dela — a
        # do próprio Drive&Act mede 30,1mm nesta métrica.
        'coerencia_osso_mm': (round(float(np.median(coerencia)), 2)
                              if coerencia else None),
        # Invariante a escala: divide cada osso pelo tronco do mesmo quadro.
        # O absoluto acima ainda carrega a deriva de escala, que é ambiguidade
        # legítima do lifting monocular e não erro do modelo.
        'coerencia_forma': (round(float(np.median(coerencia_forma)), 4)
                            if coerencia_forma else None),
        # Controle: uma pose congelada tem coerência perfeita. Sem comparar o
        # movimento, um ganho de coerência pode ser suavização disfarçada.
        'movimento_mm': (round(float(np.median(movimento)), 2)
                         if movimento else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
