#!/usr/bin/env python
"""Painel de validação em tempo real com câmera ou vídeo.

Controller: lê argumentos, liga câmera, pipeline e painel, e trata a entrada do
usuário. Nenhuma regra de estimação ou de desenho vive aqui.

Sem argumentos, usa o modelo corrente do projeto com a webcam padrão:

    python scripts/run_panel.py

Os demais argumentos servem para comparar modelos ou reproduzir um vídeo.
`--lift-ckpt ""` desliga o painel 3D, o que é útil para isolar o custo do
Módulo 3 ao medir a taxa de quadros.

**Não rode com um treino em andamento.** A disputa pela GPU derruba a taxa para
cerca de um terço da real: medidos 63ms por quadro sob contenção contra 20ms com
a GPU livre, e o número exibido no painel induziria a erro numa demonstração.

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

from src.models.observability import reliable_keypoints
from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                      FullBodyPosePipeline, PersonDetector)
from src.visualization.panel import PanelState, ValidationPanel
from src.visualization.skeleton import draw_box, draw_pose

WINDOW = 'Validador de Pose 3D Full-Body'

# Janela para suavizar o FPS exibido. Sem ela o número oscila a cada frame e
# fica ilegível, sem refletir melhor o desempenho real.
FPS_WINDOW = 30

MESSAGE_DURATION_S = 2.5

# Modelo corrente do projeto. Manter aqui, e não no exemplo da docstring, é o
# que evita que a demonstração rode com um checkpoint antigo porque alguém
# copiou a linha de comando errada — já apontava para o treino descartado do
# RTMPose-m, que mede 0,51 de AP contra 0,69 deste.
POSE_CONFIG = 'configs/eval/rtmw_x_wholebody_eval.py'
POSE_CHECKPOINT = ('work_dirs/rtmw_x_gray_lora/'
                   'best_coco-wholebody_AP_epoch_5_merged.pth')
DETECTOR_CONFIG = 'configs/detectors/rtmdet_nano_person_infer.py'
DETECTOR_CHECKPOINT = ('checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-'
                       '05d8511e.pth')

# A pontuação do SimCC é a magnitude do máximo do mapa de resposta, **não** uma
# probabilidade: ela não tem teto em 1. O padrão anterior, 0,3, estava abaixo de
# qualquer valor observado, e o painel mostrava "133 de 133 keypoints
# detectados" apontado para um quarto vazio. Medido sobre 40 imagens do COCO com
# pessoa, a mediana é 2,83 e o percentil 90 é 9,16; num frame sem pessoa a
# mediana cai para 2,28 e o percentil 90 para 3,46. As distribuições se
# sobrepõem, então nenhum limiar separa perfeitamente, mas 3,0 descarta a maior
# parte da resposta espúria sem perder os keypoints de fato localizados.
DEFAULT_SCORE_THRESHOLD = 3.0

LIFT_CONFIG = 'configs/lift3d_dstformer_h3wb_16frm.py'
LIFT_CHECKPOINT = 'work_dirs/lift3d_dstformer_h3wb/best_MPJPE_whole_epoch_30.pth'

# Uma volta completa a cada ~12 segundos a 30 FPS. Mais rápido cansa a leitura,
# mais lento não chega a revelar a profundidade.
AZIMUTH_STEP_RADIANS = 0.0175


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', default=POSE_CONFIG,
                        help='Config do estimador de pose')
    parser.add_argument('--ckpt', default=POSE_CHECKPOINT,
                        help='Checkpoint do estimador')
    parser.add_argument('--det-cfg', default=DETECTOR_CONFIG,
                        help='Config do detector; vazio dispensa o estágio')
    parser.add_argument('--det-ckpt', default=DETECTOR_CHECKPOINT,
                        help='Checkpoint do detector')
    parser.add_argument('--lift-cfg', default=LIFT_CONFIG,
                        help='Config do lifting 2D para 3D')
    parser.add_argument('--lift-ckpt', default=LIFT_CHECKPOINT,
                        help='Checkpoint do lifting; vazio desliga o painel 3D')
    parser.add_argument('--source', default='0', help='Índice de câmera ou caminho de vídeo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help='Resposta mínima para contar um keypoint como '
                             'detectado. Não é probabilidade: a saída do SimCC '
                             'não tem teto em 1'),
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


def _config_without_flip_test(config_path: str, out_dir: Path) -> str:
    """Desliga o flip test, que dobra o custo da pose sem valor em operação.

    Ele executa o modelo também sobre a imagem espelhada e faz a média dos mapas
    de resposta. Isso compra precisão, que é o que a avaliação de AP mede, e
    custa metade da taxa de quadros: medidos 12,44 ms por pessoa sem ele contra
    cerca de 24,9 ms com ele. Num painel ao vivo a métrica é latência.
    """
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    cfg.model.test_cfg = dict(cfg.model.get('test_cfg', {}))
    cfg.model.test_cfg['flip_test'] = False

    patched = out_dir / 'pose_config_ao_vivo.py'
    cfg.dump(patched)
    return str(patched)


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

    lifter = None
    if args.lift_ckpt:
        print('Carregando lifting 3D...')
        from src.models.sequence_lifter import SequenceLifter
        lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device)
    else:
        print('Sem lifting: o painel 3D fica vazio.')

    print('Carregando estimador de pose...')
    pipeline = FullBodyPosePipeline(_config_without_flip_test(args.cfg, args.out_dir),
                                    args.ckpt, args.device, detector)

    capture, source_label = open_source(args.source, args.cam_width, args.cam_height)
    panel = ValidationPanel()
    state = PanelState(source_label=source_label,
                       score_threshold=args.score_thr)

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
                            # A mesma máscara do painel 3D: o sistema não pode
                            # afirmar num quadrante o que nega no outro. Uma
                            # junta que a câmera não enxerga não é desenhada em
                            # lugar nenhum.
                            trusted = reliable_keypoints(result.scores[index],
                                                         args.score_thr)
                            draw_pose(frame, result.keypoints[index],
                                      np.where(trusted, result.scores[index],
                                               0.0), args.score_thr)

                    frame_times.append(time.perf_counter() - started)
                    state.frame = frame
                    state.num_people = result.num_people
                    state.latency_ms = result.latency_ms
                    state.region_confidence = result.region_confidence(args.score_thr)
                    state.region_counts = result.region_counts(args.score_thr)
                    state.fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
                    state.frame_index += 1

                    if lifter is not None and result.num_people:
                        height, width = frame.shape[:2]
                        state.keypoints_3d = lifter(result.keypoints[0],
                                                    result.scores[0],
                                                    (width, height))
                        # O limiar sozinho deixa passar 92% das juntas que a
                        # câmera não enxerga, porque a adaptação ao domínio as
                        # tornou confiantes sem torná-las corretas. A geometria
                        # da montagem é o que separa as duas populações.
                        state.keypoints_3d_reliable = reliable_keypoints(
                            result.scores[0], args.score_thr)
                        state.lifting_warming_up = lifter.warming_up
                    else:
                        state.keypoints_3d = None
                        if lifter is not None:
                            # Sem pessoa o buffer perderia continuidade
                            # temporal; recomeçar é melhor que misturar
                            # trechos separados por uma lacuna.
                            lifter.reset()

                    # Gira a vista continuamente: numa projeção ortográfica
                    # estática a profundidade é ambígua a olho nu.
                    state.azimuth += AZIMUTH_STEP_RADIANS

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
