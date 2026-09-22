#!/usr/bin/env python
"""Mede onde o estimador 2D coloca as juntas que a câmera não enxerga.

Controller. Roda sobre uma gravação da webcam ou sobre quadros do Drive&Act,
com o mesmo estimador do painel, e resume a distribuição por junta em larguras
de ombro --- a forma em que o número serve de parâmetro para a simulação de
treino (`src/data/estimator_noise.py`).

    python scripts/measure_absent_placement.py --fonte webcam --tag mesa
    python scripts/measure_absent_placement.py --fonte driveact --tag retrovisor

A caracterização do quadril feita antes deste script produziu a correção que
recuperou o tronco no corte simulado; as pernas não transferiram, e é a
distribuição delas que falta.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import (DEFAULT_DETECTOR,
                                        DETECTOR_CHOICES)

WEBCAM_VIDEO = Path('work_dirs/panel/rec_20260912_163434.mp4')
CACHE_2D = Path('work_dirs/live_quality')
DRIVEACT_FRAMES = Path('data/processed/driveact/val')
DRIVEACT_ANNOTATIONS = Path('data/processed/driveact/'
                            'driveact_midlevel.chunks_90.split_0.val.json')

# Quadril, joelhos, tornozelos e pés: o que uma webcam de mesa corta. Da posição
# de retrovisor o quadril aparece, e sai do conjunto pela máscara de montagem.
ABSENT_JOINTS = tuple(range(11, 23))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fonte', choices=('webcam', 'driveact'),
                        default='webcam')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR,
                        choices=DETECTOR_CHOICES,
                        help='Detector de pessoas do estágio 1; o '
                             'padrão é o de operação')
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def webcam_detections(panel, args):
    """Reaproveita o cache 2D da medição de qualidade, se existir."""
    import cv2

    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline, build_person_detector)

    cache = CACHE_2D / (f'{WEBCAM_VIDEO.stem}_'
                        f'{Path(panel.POSE_CHECKPOINT).stem}.npz')
    if cache.exists():
        dados = np.load(cache)
        print(f'2D do cache: {cache}')
        tamanho = (int(dados['frame_size'][0]), int(dados['frame_size'][1]))
        return dados['keypoints'], dados['scores'], tamanho

    detector = build_person_detector(args.detector, args.device,
                                     DEFAULT_DETECTOR_SCORE)
    pipeline = FullBodyPosePipeline(
        panel._config_without_flip_test(panel.POSE_CONFIG, CACHE_2D),
        panel.POSE_CHECKPOINT, args.device, detector)

    capture = cv2.VideoCapture(str(WEBCAM_VIDEO))
    keypoints, scores, tamanho = [], [], (0, 0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tamanho = (frame.shape[1], frame.shape[0])
        resultado = pipeline(panel.to_model_domain(frame, keep_color=False))
        if resultado.num_people:
            keypoints.append(resultado.keypoints[0])
            scores.append(resultado.scores[0])
    capture.release()
    return np.stack(keypoints), np.stack(scores), tamanho


def driveact_detections(panel, args):
    """Quadros do domínio de aplicação, com a mesma pilha do painel."""
    import cv2

    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline, build_person_detector)

    detector = build_person_detector(args.detector, args.device,
                                     DEFAULT_DETECTOR_SCORE)
    pipeline = FullBodyPosePipeline(
        panel._config_without_flip_test(panel.POSE_CONFIG, CACHE_2D),
        panel.POSE_CHECKPOINT, args.device, detector)

    anotacoes = json.loads(DRIVEACT_ANNOTATIONS.read_text())
    imagens = sorted(anotacoes['images'],
                     key=lambda i: (i['file_id'], i['frame_id']))
    passo = max(1, len(imagens) // args.max_frames)

    keypoints, scores, tamanho = [], [], (0, 0)
    for imagem in imagens[::passo][:args.max_frames]:
        frame = cv2.imread(str(DRIVEACT_FRAMES / imagem['file_name']))
        if frame is None:
            continue
        tamanho = (frame.shape[1], frame.shape[0])
        resultado = pipeline(frame)
        if resultado.num_people:
            keypoints.append(resultado.keypoints[0])
            scores.append(resultado.scores[0])
    return np.stack(keypoints), np.stack(scores), tamanho


def main():
    args = parse_args()

    from src.evaluation.absent_placement import placement, summarize

    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)

    if args.fonte == 'webcam':
        keypoints, scores, tamanho = webcam_detections(panel, args)
    else:
        keypoints, scores, tamanho = driveact_detections(panel, args)
    print(f'{len(keypoints)} quadros, {tamanho[0]}x{tamanho[1]}')

    amostras = []
    for indice in range(len(keypoints)):
        medidas = placement(keypoints[indice], scores[indice], tamanho,
                            ABSENT_JOINTS)
        if medidas is not None:
            amostras.append(medidas)

    relatorio = {
        'tag': args.tag,
        'fonte': args.fonte,
        'quadros': len(amostras),
        'tamanho_do_quadro': list(tamanho),
        'pose_checkpoint': Path(panel.POSE_CHECKPOINT).name,
        'unidade': 'larguras de ombro, origem no meio dos ombros, y para baixo',
        'juntas': summarize(amostras, ABSENT_JOINTS),
    }
    out = args.out or Path(f'results/posicao_ausentes_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    nomes = {11: 'quadril E', 12: 'quadril D', 13: 'joelho E', 14: 'joelho D',
             15: 'tornozelo E', 16: 'tornozelo D', 17: 'dedao E',
             18: 'dedinho E', 19: 'calcanhar E', 20: 'dedao D',
             21: 'dedinho D', 22: 'calcanhar D'}
    print(f'{"junta":13s}{"x":>7}{"y":>7}{"y p10":>7}{"y p90":>7}'
          f'{"borda":>8}{"na borda":>10}{"sobre corpo":>13}{"resp":>7}')
    for junta, valores in relatorio['juntas'].items():
        print(f'{nomes[int(junta)]:13s}{valores["x_mediano"]:7.2f}'
              f'{valores["y_mediano"]:7.2f}{valores["y_p10"]:7.2f}'
              f'{valores["y_p90"]:7.2f}{valores["borda_mediana"]:8.2f}'
              f'{valores["fracao_junto_da_borda"]:10.2f}'
              f'{valores["fracao_sobre_corpo_visivel"]:13.2f}'
              f'{valores["resposta_mediana"]:7.2f}')
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
