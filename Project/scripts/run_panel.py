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
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.camera_calibration import load_intrinsics
from src.data.estimator_noise import UNOBSERVED_CONFIDENCE_CAP
from src.models.observability import MOUNTING_ABSENT, observed_keypoints
from src.models.detector_config import (DEFAULT_DETECTOR, DETECTOR_CHOICES,
                                        DETECTOR_NONE)
from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                      FullBodyPosePipeline,
                                      build_person_detector)
from src.models.temporal_filter import OneEuroFilter, cutoffs_by_observation
from src.visualization.panel import PanelState, ValidationPanel
from src.visualization.skeleton import draw_box, draw_pose

WINDOW = 'Validador de Pose 3D Full-Body'

# Janela para suavizar o FPS exibido. Sem ela o número oscila a cada frame e
# fica ilegível, sem refletir melhor o desempenho real.
FPS_WINDOW = 30

MESSAGE_DURATION_S = 2.5

# Checkpoint de pose por montagem, porque cada um foi medido melhor no seu
# domínio e a diferença entre eles é grande demais para um padrão só.
#
# Retrovisor: o modelo adaptado ao infravermelho com ensaio, que mede 9,94px de
# erro corporal no Drive&Act contra 15,04px do modelo de grayscale --- e que,
# ao contrário da versão sem ensaio, preserva face e mãos (whole-body AP 0,6848
# contra 0,2330 no COCO em cinza).
#
# Mesa: o modelo de grayscale da Etapa 2. Na gravação da própria webcam ele
# treme menos que o do ensaio (face 1,39 contra 3,22mm, coerência de forma
# 0,146 contra 0,174), o que faz sentido --- é o domínio em que ele foi
# treinado, enquanto o outro foi especializado no habitáculo. A anatomia do
# ensaio é melhor (tronco 348,7 contra 317,3mm), e essa troca se decide pela
# montagem, não por preferência.
#
# **Não usar o checkpoint sem ensaio (`rtmw_x_driveact_ft_v2`) em montagem
# alguma**: na webcam a face dele explode para 164,5px de raio contra 35,2 dos
# outros dois, e a distância interpupilar reconstruída sai em 225mm.
POSE_CONFIG = 'configs/eval/rtmw_x_wholebody_eval.py'
POSE_CHECKPOINT_BY_MOUNTING = {
    'mesa': ('work_dirs/rtmw_x_gray_lora/'
             'best_coco-wholebody_AP_epoch_5_merged.pth'),
    'retrovisor': ('work_dirs/rtmw_x_driveact_ensaio/'
                   'best_torso_px_mean_epoch_2_merged.pth'),
}

# Checkpoint das medições já publicadas em `results/`, que os scripts de
# medição importam. Mantê-lo explícito evita que uma troca de padrão do painel
# mude em silêncio o protocolo de um número que o documento cita.
POSE_CHECKPOINT = POSE_CHECKPOINT_BY_MOUNTING['mesa']

# A pontuação do SimCC é a magnitude do máximo do mapa de resposta, **não** uma
# probabilidade: ela não tem teto em 1. O padrão anterior, 0,3, estava abaixo de
# qualquer valor observado, e o painel mostrava "133 de 133 keypoints
# detectados" apontado para um quarto vazio. Medido sobre 40 imagens do COCO com
# pessoa, a mediana é 2,83 e o percentil 90 é 9,16; num frame sem pessoa a
# mediana cai para 2,28 e o percentil 90 para 3,46. As distribuições se
# sobrepõem, então nenhum limiar separa perfeitamente, mas 3,0 descarta a maior
# parte da resposta espúria sem perder os keypoints de fato localizados.
DEFAULT_SCORE_THRESHOLD = 3.0

# Lifting por montagem, pelo mesmo motivo do estimador de pose: cada um foi
# medido melhor no seu domínio, e aqui a diferença é grande demais para um
# padrão só.
#
# Retrovisor: o adaptado ao domínio veicular, treinado com o 2D do estimador
# real e a referência 3D do Drive&Act. Mede 41,5mm de PA-MPJPE contra 81,8 do
# outro, erro absoluto 60,7 contra 183,7, e coerência de osso 19,6 contra 25,2
# **com mais movimento** --- ou seja, o ganho não é suavização.
#
# Mesa: o treinado com corte de quadro. O veicular **quebra** fora da montagem
# dele: na gravação da webcam a canela predita sai em 18,9mm e a distância
# interpupilar em 35,8, contra 297,5 e 62,1 deste. Especializou-se numa câmera,
# um enquadramento e oito sujeitos.
LIFT_CONFIG_BY_MOUNTING = {
    'mesa': 'configs/lift3d_dstformer_h3wb_robusto_v3.py',
    'retrovisor': 'configs/lift3d_veicular.py',
}
LIFT_CHECKPOINT_BY_MOUNTING = {
    'mesa': 'work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth',
    # O retreinado com a perda que respeita o peso do alvo. O primeiro treino
    # veicular (work_dirs/lift3d_veicular) colapsava face, mãos e pernas ---
    # interpupilar reconstruída de 3mm no próprio Drive&Act, contra 71mm deste.
    'retrovisor': ('work_dirs/lift3d_veicular_peso/'
                   'best_MPJPE_whole_epoch_5.pth'),
}

# Config e checkpoint das medições já publicadas, que os scripts de medição
# importam. Mantê-los explícitos evita que uma troca de padrão do painel mude
# em silêncio o protocolo de um número que o documento cita.
LIFT_CONFIG = LIFT_CONFIG_BY_MOUNTING['mesa']
LIFT_CHECKPOINT = LIFT_CHECKPOINT_BY_MOUNTING['mesa']

# Teto de confiança das juntas que o sistema sabe não ter observado. É contrato
# entre treino e inferência, e por isso anda junto do checkpoint: só vale para
# quem treinou com ele. Aplicá-lo ao v2, que não treinou, piora o quadril de
# 495,8 para 609,9mm no mesmo protocolo. `None` desliga.
LIFT_UNOBSERVED_CONFIDENCE = UNOBSERVED_CONFIDENCE_CAP

# Calibração da câmera própria, produzida por scripts/calibrate_camera.py. Sem
# ela a escala da pose 3D é herdada de outro dataset, o que no Drive&Act ampliou
# a pose em 3,8 vezes — e o painel declara a diferença em vez de escondê-la.
CAMERA_CALIBRATION = 'configs/camera/webcam.calibration.json'

# Distância típica da câmera ao ocupante, por montagem. É a única grandeza que a
# câmera monocular não observa, e dela dependem tanto a escala da entrada da rede
# quanto a da saída.
#
# Mesa: 1,17m medidos com trena da lente ao rosto em 21/09/2026. O valor
# substitui a estimativa antropométrica de 1,35m que vigorava antes, e desloca
# uma questão em aberto: com 1,17m a distância interpupilar reconstruída na
# gravação de 12/09 sai em cerca de 50mm, impossível num adulto. Ou aquela
# gravação foi feita a outra distância, ou o lifting subdimensiona a pose em
# torno de 20%. A medição que decide é uma gravação nova à distância medida.
# Retrovisor: 0,66m, mediana medida na referência 3D do Drive&Act.
DEFAULT_SUBJECT_DEPTH_M = {'mesa': 1.17, 'retrovisor': 0.664}

# Precisão da inferência do lifting, o estágio que domina o caminho completo.
# Medido (scripts/benchmark_lifting.py, v3, 300 janelas do S7): float32 27,91ms
# e 39,31mm; float16 10,72ms e 39,35mm; bfloat16 11,72ms e 39,38mm. float16 corta
# a latência 2,6 vezes por 0,04mm.
LIFT_PRECISIONS = ('float32', 'float16', 'bfloat16')
DEFAULT_LIFT_PRECISION = 'float16'

# De quantos em quantos quadros a janela do lifting é montada. 1 até que a
# comparação com 3 (scripts/run_passo_temporal.sh) decida.
DEFAULT_TEMPORAL_STRIDE = 1

# Taxa nominal do painel, que o One Euro assume constante. Um desvio de alguns
# hertz desloca o corte efetivo na mesma proporção, sem quebrar o filtro.
FILTER_RATE_HZ = 30.0

# Balanço da vista 3D, em lugar da volta completa que havia aqui. Uma volta
# contínua tira a referência de frente justo quando se quer conferir a pose;
# ±25° dão paralaxe suficiente para separar em profundidade dois membros
# sobrepostos mantendo o corpo de frente. O período em quadros, e não em
# segundos, mantém o balanço idêntico entre a câmera e a reprodução de um vídeo;
# 180 quadros são cerca de seis segundos a 30 FPS.
SWAY_AMPLITUDE_RADIANS = np.deg2rad(25.0)
SWAY_PERIOD_FRAMES = 180


def confianca_opcional(texto: str) -> float | None:
    """Lê o teto de confiança, aceitando "nenhum" para desligá-lo.

    Existe porque o teto não é um número sempre presente: ele é contrato com o
    treino do checkpoint, e um checkpoint que não o viu precisa recebê-lo
    ausente, não zerado --- zero é uma confiança, ausência não é.
    """
    if texto.strip().lower() in ('', 'nenhum', 'none'):
        return None
    return float(texto)


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
    parser.add_argument('--teto-confianca', type=confianca_opcional,
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


def lift_dtype(precision: str):
    """O tipo do PyTorch para a precisão pedida; `None` mantém float32."""
    import torch
    return None if precision == 'float32' else getattr(torch, precision)


def build_lifter(args):
    """Monta o lifting 3D com a geometria da câmera, se ela for conhecida.

    Returns:
        (lifter, calibrado). `lifter` é `None` quando o painel 3D está desligado.
    """
    if not args.lift_ckpt:
        print('Sem lifting: o painel 3D fica vazio.')
        return None, False

    print('Carregando lifting 3D...')
    from src.models.sequence_lifter import CameraView, SequenceLifter

    intrinsics = load_intrinsics(args.calibracao)
    camera = None
    if intrinsics is not None:
        camera = CameraView(focal_length_px=intrinsics.fx,
                            principal_point=(intrinsics.cx, intrinsics.cy),
                            subject_depth_m=args.distancia)
        print(f'  calibração: fx {intrinsics.fx:.1f}px, centro '
              f'({intrinsics.cx:.0f}, {intrinsics.cy:.0f}), ocupante a '
              f'{args.distancia:.2f}m')
    else:
        print(f'  sem calibração em {args.calibracao}; escala herdada do H3WB, '
              f'pose aproximada em tamanho')

    lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device,
                            camera=camera,
                            unobserved_confidence=args.teto_confianca,
                            frame_stride=args.passo_temporal,
                            inference_dtype=lift_dtype(args.precisao))
    return lifter, camera is not None


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
    pipeline = FullBodyPosePipeline(_config_without_flip_test(args.cfg, args.out_dir),
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

                    frame_times.append(time.perf_counter() - started)
                    state.frame = frame
                    state.num_people = result.num_people
                    state.latency_ms = result.latency_ms
                    state.region_confidence = result.region_confidence(
                        args.score_thr, observed)
                    state.region_counts = result.region_counts(
                        args.score_thr, observed)
                    state.fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
                    state.frame_index += 1

                    if lifter is not None and result.num_people:
                        # Filtro por junta: o que foi observado segue
                        # responsivo, o que é previsão é estabilizado. Sem
                        # isso a perna prevista treme 213mm por quadro e é o
                        # que o olho lê como "não está pegando".
                        corte, ganho = cutoffs_by_observation(observed[0])
                        state.keypoints_3d = smoother(
                            lifter(result.keypoints[0], result.scores[0],
                                   (width, height), observed=observed[0]),
                            corte, ganho)
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
