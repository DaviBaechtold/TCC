#!/usr/bin/env python
"""Painel de validação em tempo real com câmera ou vídeo.

Controller: lê argumentos, liga câmera, pipeline e painel, e trata a entrada do
usuário. Nenhuma regra de estimação ou de desenho vive aqui.

Sem argumentos, usa o modelo corrente do projeto com a webcam de mesa:

    python scripts/run_panel.py

Para um vídeo do Drive&Act, a montagem muda o que o sistema considera
observável:

    python scripts/run_panel.py --source video.mp4 --montagem retrovisor \\
        --calibracao run.calibration.json --distancia 0.66

Os demais argumentos servem para comparar modelos. `--lift-ckpt ""` desliga o
painel 3D, o que é útil para isolar o custo do Módulo 3 ao medir a taxa de
quadros.

**Não rode com um treino em andamento.** A disputa pela GPU derruba a taxa para
cerca de um terço da real: medidos 63ms por quadro sob contenção contra 20ms com
a GPU livre, e o número exibido no painel induziria a erro numa demonstração.

Teclas: espaço pausa, r grava, s salva frame, k alterna esqueleto, q sai.
"""

import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.observability import MOUNTING_ABSENT, observed_keypoints
from src.models.detector_config import (DEFAULT_DETECTOR, DETECTOR_CHOICES,
                                        DETECTOR_NONE)
from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                      FullBodyPosePipeline,
                                      build_person_detector)
from src.models.operating_config import (
    CAMERA_CALIBRATION, DEFAULT_LIFT_PRECISION, DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SUBJECT_DEPTH_M, DEFAULT_TEMPORAL_STRIDE, FILTER_RATE_HZ,
    LIFT_CHECKPOINT_BY_MOUNTING, LIFT_CONFIG_BY_MOUNTING, LIFT_PRECISIONS,
    LIFT_UNOBSERVED_CONFIDENCE, POSE_CHECKPOINT_BY_MOUNTING, POSE_CONFIG,
    build_lifter, config_without_flip_test, optional_confidence,
    to_model_domain)
from src.models.temporal_filter import OneEuroFilter, cutoffs_by_observation
from src.visualization.panel import PanelState, ValidationPanel
from src.visualization.skeleton import draw_box, draw_pose

WINDOW = 'Validador de Pose 3D Full-Body'

# Janela para suavizar o FPS exibido. Sem ela o número oscila a cada frame e
# fica ilegível, sem refletir melhor o desempenho real.
FPS_WINDOW = 30

MESSAGE_DURATION_S = 2.5

# Taxa pedida à webcam. Sem pedido explícito a C922 negocia 10 FPS em 1280x720,
# e o painel exibia 42 FPS enquanto processava 10 --- medido em 24/09/2026, 439
# quadros gravados em 44 segundos.
CAMERA_FPS = 30

# Balanço da vista 3D, em lugar da volta completa que havia aqui. Uma volta
# contínua tira a referência de frente justo quando se quer conferir a pose;
# ±25° dão paralaxe suficiente para separar em profundidade dois membros
# sobrepostos mantendo o corpo de frente. O período em quadros, e não em
# segundos, mantém o balanço idêntico entre a câmera e a reprodução de um vídeo;
# 180 quadros são cerca de seis segundos a 30 FPS.
SWAY_AMPLITUDE_RADIANS = np.deg2rad(25.0)
SWAY_PERIOD_FRAMES = 180


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', default=POSE_CONFIG,
                        help='Config do estimador de pose')
    parser.add_argument('--ckpt', default=None,
                        help='Checkpoint do estimador')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR,
                        choices=DETECTOR_CHOICES,
                        help='Detector de pessoas do estágio 1')
    parser.add_argument('--lift-cfg', default=None,
                        help='Config do lifting 2D para 3D')
    parser.add_argument('--lift-ckpt', default=None,
                        help='Checkpoint do lifting; vazio desliga o painel 3D')
    parser.add_argument('--teto-confianca', type=optional_confidence,
                        default=LIFT_UNOBSERVED_CONFIDENCE,
                        help='Teto de confiança das juntas não '
                             'observadas. Precisa casar com o que o '
                             'checkpoint de lifting viu no treino')
    parser.add_argument('--montagem', default='mesa',
                        choices=sorted(MOUNTING_ABSENT),
                        help='Onde a câmera está montada. Decide quais juntas '
                             'não aparecem em quadro algum: no retrovisor, '
                             'joelhos, tornozelos e pés')
    parser.add_argument('--calibracao', default=CAMERA_CALIBRATION,
                        help='Calibração da câmera. Sem ela a escala da pose 3D '
                             'é herdada e apenas aproximada')
    parser.add_argument('--precisao', default=DEFAULT_LIFT_PRECISION,
                        choices=LIFT_PRECISIONS,
                        help='Precisão da inferência do lifting')
    parser.add_argument('--passo-temporal', type=int,
                        default=DEFAULT_TEMPORAL_STRIDE,
                        help='De quantos em quantos quadros a janela do lifting '
                             'é montada. O H3WB tem 100ms medianos entre quadros '
                             'da janela; a 30 FPS, passo 3 reproduz isso')
    parser.add_argument('--distancia', type=float, default=None,
                        help='Distância da câmera ao ocupante, em metros. '
                             'Medir uma vez: é o que a câmera não observa. '
                             'O padrão depende da montagem')
    parser.add_argument('--source', default='0', help='Índice de câmera ou caminho de vídeo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help='Resposta mínima para contar um keypoint como '
                             'detectado. Não é probabilidade: a saída do SimCC '
                             'não tem teto em 1')
    parser.add_argument('--bbox-thr', type=float,
                        default=DEFAULT_DETECTOR_SCORE,
                        help='Confiança mínima do detector de pessoas; '
                             'calibrada por medição em src/models/pose_pipeline.py')
    parser.add_argument('--cam-width', type=int, default=1280)
    parser.add_argument('--cam-height', type=int, default=720)
    parser.add_argument('--color', action='store_true',
                        help='Mantém a imagem colorida. Por padrão converte para '
                             'escala de cinza, que é o domínio em que o modelo '
                             'foi treinado como proxy de infravermelho')
    parser.add_argument('--out-dir', type=Path, default=Path('work_dirs/panel'))

    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = POSE_CHECKPOINT_BY_MOUNTING[args.montagem]
    if args.lift_cfg is None:
        args.lift_cfg = LIFT_CONFIG_BY_MOUNTING[args.montagem]
    if args.lift_ckpt is None:
        args.lift_ckpt = LIFT_CHECKPOINT_BY_MOUNTING[args.montagem]
    if args.distancia is None:
        args.distancia = DEFAULT_SUBJECT_DEPTH_M[args.montagem]
    return args


def open_source(source: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # MJPG evita o teto de ~5 FPS que o formato YUYV impõe em 1280x720
        # nas webcams USB, que estrangularia a medição de FPS do sistema.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _fix_camera_framerate(int(source))
        label = f'Camera {source}'
    else:
        capture = cv2.VideoCapture(source)
        label = Path(source).name

    if not capture.isOpened():
        raise SystemExit(f'Nao foi possivel abrir a fonte: {source}')
    return capture, label


def _fix_camera_framerate(index: int) -> None:
    """Impede a webcam de baixar a taxa de quadros para alongar a exposição.

    Com a exposição automática, o controle `exposure_dynamic_framerate` deixa a
    câmera trocar quadros por luz: num quarto iluminado por janela ela entrega
    15 FPS mesmo com 30 pedidos, e 30 com o controle desligado, sem imagem
    escura (brilho médio 110). O OpenCV não expõe esse controle; o v4l2-ctl sim.
    """
    try:
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{index}',
                        '-c', 'exposure_dynamic_framerate=0'],
                       check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print('  aviso: taxa dinamica da camera nao desligada (v4l2-ctl); '
              'com pouca luz a camera pode entregar menos de 30 FPS')


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.detector == DETECTOR_NONE:
        print('Sem detector: o frame inteiro e usado como regiao de interesse.')
    else:
        print(f'Carregando detector de pessoas ({args.detector})...')
    detector = build_person_detector(args.detector, args.device, args.bbox_thr)

    lifter, calibrado = build_lifter(args)
    # O One Euro trabalha na saída 3D, em metros. Medido: filtrar ali reduz 69%
    # do tremor a 0,14 quadro de atraso, e filtrar o 2D de entrada rende menos,
    # porque a rede é temporal e espalha o ruído pela janela de dezesseis.
    smoother = OneEuroFilter(FILTER_RATE_HZ)

    print('Carregando estimador de pose...')
    pipeline = FullBodyPosePipeline(config_without_flip_test(args.cfg, args.out_dir),
                                    args.ckpt, args.device, detector)

    capture, source_label = open_source(args.source, args.cam_width, args.cam_height)
    panel = ValidationPanel()
    state = PanelState(source_label=source_label,
                       score_threshold=args.score_thr,
                       calibrated=calibrado,
                       subject_depth_m=args.distancia)

    clicked: dict[str, str | None] = {'key': None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            button = panel.button_at(x, y)
            if button is not None:
                clicked['key'] = button.key

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
    show_skeleton = True
    writer = None
    last_frame_at = None
    message_until = 0.0
    last_render = None

    try:
        while True:
            if not state.paused:
                ok, raw = capture.read()
                if not ok:
                    state.message = 'Fim do video'
                    state.paused = True
                else:
                    frame = to_model_domain(raw, args.color)
                    # A gravação serve de entrada para as medições ao vivo, que
                    # rodam o pipeline de novo sobre ela; com o esqueleto
                    # desenhado por cima, o estimador leria as próprias linhas.
                    if writer is not None:
                        writer.write(frame)
                    result = pipeline(frame)

                    height, width = frame.shape[:2]
                    # Uma única máscara por quadro alimenta o overlay 2D, as
                    # métricas por região, a confiança que o lifting recebe e o
                    # traço do painel 3D. Calculá-la em cada lugar deixaria o
                    # sistema afirmando num quadrante o que nega no outro — um
                    # print do painel exibia "Pes 2/6" sem um pé desenhado.
                    observed = observed_keypoints(
                        result.keypoints, result.scores, (width, height),
                        args.score_thr, args.montagem)

                    if show_skeleton:
                        for index in range(result.num_people):
                            draw_box(frame, result.boxes[index])
                            draw_pose(frame, result.keypoints[index],
                                      np.where(observed[index],
                                               result.scores[index], 0.0),
                                      args.score_thr)

                    state.frame = frame
                    state.num_people = result.num_people
                    state.latency_ms = dict(result.latency_ms)
                    state.region_confidence = result.region_confidence(
                        args.score_thr, observed)
                    state.region_counts = result.region_counts(
                        args.score_thr, observed)
                    state.frame_index += 1

                    if lifter is not None and result.num_people:
                        # Filtro por junta: o que foi observado segue
                        # responsivo, o que é previsão é estabilizado. Sem
                        # isso a perna prevista treme 213mm por quadro e é o
                        # que o olho lê como "não está pegando".
                        corte, ganho = cutoffs_by_observation(observed[0])
                        lift_started = time.perf_counter()
                        pose_3d = lifter(result.keypoints[0], result.scores[0],
                                         (width, height), observed=observed[0])
                        state.latency_ms['lift'] = \
                            (time.perf_counter() - lift_started) * 1e3
                        state.keypoints_3d = smoother(pose_3d, corte, ganho)
                        state.keypoints_3d_observed = observed[0]
                        state.lifting_warming_up = lifter.warming_up
                    else:
                        state.keypoints_3d = None
                        if lifter is not None:
                            # Sem pessoa o buffer perderia continuidade
                            # temporal; recomeçar é melhor que misturar
                            # trechos separados por uma lacuna. O filtro
                            # acompanha, ou carregaria a pose antiga.
                            lifter.reset()
                            smoother.reset()

                    state.azimuth = SWAY_AMPLITUDE_RADIANS * np.sin(
                        2 * np.pi * state.frame_index / SWAY_PERIOD_FRAMES)

                    # FPS de relógio, de um quadro ao seguinte: inclui câmera,
                    # lifting e desenho. Medir só detector e pose exibia 42 FPS
                    # num painel que rodava a 10.
                    now = time.perf_counter()
                    if last_frame_at is not None:
                        frame_times.append(now - last_frame_at)
                    last_frame_at = now
                    state.fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            if time.time() > message_until:
                state.message = '' if not state.paused else state.message

            last_render = panel.render(state)
            cv2.imshow(WINDOW, last_render)

            pressed = cv2.waitKey(1) & 0xFF
            action = clicked['key']
            clicked['key'] = None
            if pressed == ord(' '):
                action = 'space'
            elif pressed in (ord('q'), 27):
                action = 'q'
            elif pressed in (ord('r'), ord('s'), ord('k')):
                action = chr(pressed)

            if action == 'q':
                break
            if action == 'space':
                state.paused = not state.paused
                last_frame_at = None
            elif action == 'k':
                show_skeleton = not show_skeleton
                next(b for b in panel.buttons if b.key == 'k').active = show_skeleton
            elif action == 's' and state.frame is not None:
                path = args.out_dir / f'frame_{datetime.now():%Y%m%d_%H%M%S}.png'
                cv2.imwrite(str(path), last_render)
                state.message = f'Salvo: {path.name}'
                message_until = time.time() + MESSAGE_DURATION_S
            elif action == 'r':
                if writer is None and state.frame is not None:
                    path = args.out_dir / f'rec_{datetime.now():%Y%m%d_%H%M%S}.mp4'
                    height, width = state.frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                        CAMERA_FPS, (width, height))
                    state.message = f'Gravando: {path.name}'
                else:
                    writer.release()
                    writer = None
                    state.message = 'Gravacao encerrada'
                message_until = time.time() + MESSAGE_DURATION_S
                next(b for b in panel.buttons if b.key == 'r').active = writer is not None
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
