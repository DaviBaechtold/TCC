#!/usr/bin/env python
"""Mede a qualidade da pose 3D sobre um vídeo, com o mesmo caminho do painel.

Controller. Existe porque as duas queixas que motivaram esta rodada de trabalho
--- "muito ruído" e "não gera o modelo de corpo inteiro" --- só apareciam na
inspeção visual, e inspeção visual não entra em tabela. O que ele mede está em
`src/evaluation/live_quality.py`; aqui só se lê argumento, roda o pipeline e
grava.

O passo 2D é caro e não depende do checkpoint de lifting sob teste, então fica
em cache num `.npz` ao lado do resultado: comparar três liftings sobre a mesma
gravação roda o estimador uma vez só.

    python scripts/measure_live_quality.py --tag v3 \\
        --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v3.py \\
        --lift-ckpt work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth

**Não rode com um treino em andamento**: a disputa pela GPU não muda a pose,
mas muda a taxa, e a taxa também é relatada aqui.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_VIDEO = Path('work_dirs/panel/rec_20260912_163434.mp4')
CACHE_DIR = Path('work_dirs/live_quality')

# Quadros de aquecimento da janela temporal: a saída deles tem contexto
# artificial (a janela é preenchida por repetição) e não representa o regime.
WARMUP_FRAMES = 16


def load_panel():
    """O painel é a definição operacional do sistema; medir outra coisa não vale."""
    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    return panel


def parse_args(panel):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cfg', default=panel.POSE_CONFIG)
    parser.add_argument('--ckpt', default=panel.POSE_CHECKPOINT)
    parser.add_argument('--lift-cfg', default=panel.LIFT_CONFIG)
    parser.add_argument('--lift-ckpt', default=panel.LIFT_CHECKPOINT)
    parser.add_argument('--calibracao', default=panel.CAMERA_CALIBRATION)
    parser.add_argument('--montagem', default='mesa',
                        choices=sorted(panel.DEFAULT_SUBJECT_DEPTH_M))
    parser.add_argument('--distancia', type=float, default=None,
                        help='Distância ao ocupante em metros; o padrão vem da '
                             'montagem')
    parser.add_argument('--teto-confianca', type=panel.confianca_opcional,
                        default=panel.LIFT_UNOBSERVED_CONFIDENCE,
                        help='Teto de confiança das juntas não observadas; '
                             'precisa casar com o treino do checkpoint')
    parser.add_argument('--score-thr', type=float,
                        default=panel.DEFAULT_SCORE_THRESHOLD)
    parser.add_argument('--bbox-thr', type=float, default=None)
    parser.add_argument('--sem-filtro', action='store_true',
                        help='Desliga o filtro temporal, para medir o que ele '
                             'contribui')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    if args.distancia is None:
        args.distancia = panel.DEFAULT_SUBJECT_DEPTH_M[args.montagem]
    return args


def detect_2d(panel, args) -> dict:
    """Roda detector e estimador sobre o vídeo, com cache em disco."""
    import cv2

    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline, PersonDetector)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'{args.video.stem}_{Path(args.ckpt).stem}.npz'
    if cache.exists():
        dados = np.load(cache)
        print(f'2D do cache: {cache}')
        return {chave: dados[chave] for chave in dados.files}

    detector = PersonDetector(panel.DETECTOR_CONFIG, panel.DETECTOR_CHECKPOINT,
                              args.device,
                              args.bbox_thr or DEFAULT_DETECTOR_SCORE)
    pipeline = FullBodyPosePipeline(
        panel._config_without_flip_test(args.cfg, CACHE_DIR),
        args.ckpt, args.device, detector)

    capture = cv2.VideoCapture(str(args.video))
    keypoints, scores, largura, altura = [], [], 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        altura, largura = frame.shape[:2]
        resultado = pipeline(panel.to_model_domain(frame, keep_color=False))
        if not resultado.num_people:      # sem pessoa, sem pose a medir
            continue
        keypoints.append(resultado.keypoints[0])
        scores.append(resultado.scores[0])
    capture.release()

    dados = {'keypoints': np.stack(keypoints), 'scores': np.stack(scores),
             'frame_size': np.array([largura, altura])}
    np.savez_compressed(cache, **dados)
    print(f'2D gravado em {cache}: {len(keypoints)} quadros')
    return dados


def main():
    panel = load_panel()
    args = parse_args(panel)

    from src.evaluation.live_quality import report
    from src.models.observability import observed_keypoints
    from src.models.temporal_filter import OneEuroFilter, cutoffs_by_observation

    dados = detect_2d(panel, args)
    keypoints, scores = dados['keypoints'], dados['scores']
    frame_size = (int(dados['frame_size'][0]), int(dados['frame_size'][1]))

    lifter, calibrado = panel.build_lifter(args)
    smoother = OneEuroFilter(panel.FILTER_RATE_HZ)

    poses, observados = [], []
    for indice in range(len(keypoints)):
        observado = observed_keypoints(keypoints[indice], scores[indice],
                                       frame_size, args.score_thr,
                                       args.montagem)
        pose = lifter(keypoints[indice], scores[indice], frame_size, observado)
        if not args.sem_filtro:
            corte, ganho = cutoffs_by_observation(observado)
            pose = smoother(pose, corte, ganho)
        poses.append(pose)
        observados.append(observado)

    poses = np.stack(poses)[WARMUP_FRAMES:]
    observados = np.stack(observados)[WARMUP_FRAMES:]

    # A pose fica em disco junto do relatório: comparar ajustes do filtro não
    # deve exigir reprocessar o vídeo inteiro, e a sequência é o insumo de
    # qualquer análise posterior.
    np.savez_compressed(CACHE_DIR / f'poses_{args.tag}.npz',
                        poses=poses, observados=observados)

    relatorio = report(poses, observados)
    relatorio.update({
        'tag': args.tag,
        'video': args.video.name,
        'lift_checkpoint': Path(args.lift_ckpt).name,
        'pose_checkpoint': Path(args.ckpt).name,
        'montagem': args.montagem,
        'distancia_m': args.distancia,
        'calibrado': bool(calibrado),
        'filtro': not args.sem_filtro,
    })

    out = args.out or Path(f'results/qualidade_ao_vivo_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    print(f'\ntremor  {relatorio["tremor_mm"]}')
    print(f'movimento {relatorio["movimento_mm"]}mm  '
          f'coerência {relatorio["coerencia_osso_mm"]}mm')
    for nome, valores in relatorio['geometria'].items():
        marca = 'ok ' if valores['dentro_da_faixa'] else 'FORA'
        print(f'  {marca} {nome:20s} {valores["mediana_mm"]:7.1f}mm  '
              f'faixa {valores["faixa_adulto_mm"]}')
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
