#!/usr/bin/env python
"""Painel de validação em tempo real com câmera ou vídeo.

Controller: lê argumentos, liga câmera, pipeline e painel, e trata a entrada do
usuário. Nenhuma regra de estimação ou de desenho vive aqui.

Exemplo:
    python scripts/run_panel.py \\
        --cfg work_dirs/ft_smoke/rtmpose_m_wholebody_gray_ft.py \\
        --ckpt work_dirs/ft_smoke/best_coco-wholebody_AP_epoch_10.pth \\
        --det-cfg configs/detectors/rtmdet_nano_person_infer.py \\
        --det-ckpt checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \\
        --source 0

Teclas: espaço pausa, r grava, s salva frame, k alterna esqueleto, q sai.
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pose_pipeline import FullBodyPosePipeline, PersonDetector
from src.visualization.panel import PanelState, ValidationPanel
from src.visualization.skeleton import draw_box, draw_pose

WINDOW = 'Validador de Pose 3D Full-Body'

# Janela para suavizar o FPS exibido. Sem ela o número oscila a cada frame e
# fica ilegível, sem refletir melhor o desempenho real.
FPS_WINDOW = 30

MESSAGE_DURATION_S = 2.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', required=True, help='Config do estimador de pose')
    parser.add_argument('--ckpt', required=True, help='Checkpoint do estimador')
    parser.add_argument('--det-cfg', default='', help='Config do detector (opcional)')
    parser.add_argument('--det-ckpt', default='', help='Checkpoint do detector')
    parser.add_argument('--source', default='0', help='Índice de câmera ou caminho de vídeo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Confiança mínima para desenhar um keypoint')
    parser.add_argument('--bbox-thr', type=float, default=0.3,
                        help='Confiança mínima do detector de pessoas. Medido '
                             'em 40 imagens do COCO val: em 0.3 o detector '
                             'produz 84 caixas para 73 pessoas anotadas, '
                             'enquanto em 0.5 recupera apenas 60% delas')
    parser.add_argument('--cam-width', type=int, default=1280)
    parser.add_argument('--cam-height', type=int, default=720)
    parser.add_argument('--color', action='store_true',
                        help='Mantém a imagem colorida. Por padrão converte para '
                             'escala de cinza, que é o domínio em que o modelo '
                             'foi treinado como proxy de infravermelho')
    parser.add_argument('--out-dir', type=Path, default=Path('work_dirs/panel'))
    return parser.parse_args()


def open_source(source: str, width: int, height: int) -> tuple[cv2.VideoCapture, str]:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # MJPG evita o teto de ~5 FPS que o formato YUYV impõe em 1280x720
        # nas webcams USB, que estrangularia a medição de FPS do sistema.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        label = f'Camera {source}'
    else:
        capture = cv2.VideoCapture(source)
        label = Path(source).name

    if not capture.isOpened():
        raise SystemExit(f'Nao foi possivel abrir a fonte: {source}')
    return capture, label


def to_model_domain(frame: np.ndarray, keep_color: bool) -> np.ndarray:
    """Converte o frame para o domínio em que o modelo foi treinado.

    O modelo é treinado em COCO-WholeBody convertido para escala de cinza, como
    proxy do infravermelho. Alimentá-lo com RGB o colocaria fora do domínio de
    treino e mediria outra coisa que não o sistema proposto. Os três canais são
    mantidos porque o backbone pré-treinado espera essa forma de entrada.
    """
    if keep_color:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.det_cfg and args.det_ckpt:
        print('Carregando detector de pessoas...')
        detector = PersonDetector(args.det_cfg, args.det_ckpt,
                                  args.device, args.bbox_thr)
    else:
        print('Sem detector: o frame inteiro e usado como regiao de interesse.')

    print('Carregando estimador de pose...')
    pipeline = FullBodyPosePipeline(args.cfg, args.ckpt, args.device, detector)

    capture, source_label = open_source(args.source, args.cam_width, args.cam_height)
    panel = ValidationPanel()
    state = PanelState(source_label=source_label)

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
                    started = time.perf_counter()
                    frame = to_model_domain(raw, args.color)
                    result = pipeline(frame)

                    if show_skeleton:
                        for index in range(result.num_people):
                            draw_box(frame, result.boxes[index])
                            draw_pose(frame, result.keypoints[index],
                                      result.scores[index], args.score_thr)

                    frame_times.append(time.perf_counter() - started)
                    state.frame = frame
                    state.num_people = result.num_people
                    state.latency_ms = result.latency_ms
                    state.region_confidence = result.region_confidence(args.score_thr)
                    state.region_counts = result.region_counts(args.score_thr)
                    state.fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
                    state.frame_index += 1

                    if writer is not None:
                        writer.write(frame)

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
                        max(1.0, state.fps), (width, height))
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
