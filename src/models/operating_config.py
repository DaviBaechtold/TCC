"""A configuração de operação do sistema: que modelo roda em cada montagem.

Camada Model. O painel, a bateria de validação e os scripts de medição precisam
concordar sobre o que é "o sistema" --- que checkpoint, que limiar, que precisão,
a que distância --- ou cada um mediria um sistema diferente. Isso morava no
controller do painel, e os scripts o carregavam por `importlib` para ler
constantes: regra de negócio no controller, e um controller importando outro.

Cada escolha aqui foi decidida por medição, e o comentário ao lado registra qual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.camera_calibration import load_intrinsics
from src.data.estimator_noise import UNOBSERVED_CONFIDENCE_CAP

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
# medição usam. Mantê-lo explícito evita que uma troca de padrão do painel
# mude em silêncio o protocolo de um número que o documento cita.
POSE_CHECKPOINT = POSE_CHECKPOINT_BY_MOUNTING['mesa']

# A pontuação do SimCC é a magnitude do máximo do mapa de resposta, **não** uma
# probabilidade: ela não tem teto em 1. O padrão anterior, 0,3, estava abaixo de
# qualquer valor observado, e o painel mostrava "133 de 133 keypoints
# detectados" apontado para um quarto vazio. Medido sobre 40 imagens do COCO com
# pessoa, a mediana é 2,83 e o percentil 90 é 9,16; num frame sem pessoa a
# mediana cai para 2,28 e o percentil 90 para 3,46. As distribuições se
# sobrepõem, então nenhum limiar separa perfeitamente, mas 3,0 descarta a maior
# parte da resposta espúria sem perder os keypoints de fato localizados. Para a
# junta oculta ele é fraco --- 67% dos tornozelos não anotados do COCO passam
# dele ---, e é por isso que `src/models/observability.py` não depende só dele.
DEFAULT_SCORE_THRESHOLD = 3.0

# Lifting por montagem, pelo mesmo motivo do estimador de pose.
#
# Retrovisor: o adaptado ao domínio veicular, treinado com o 2D do estimador
# real e a referência 3D do Drive&Act, com a perda que respeita o peso do alvo.
# Nos 12 pontos corporais mede 38,34mm de PA-MPJPE contra 81,79 do modelo sem
# adaptação, e erro absoluto de 57,38 contra 183,71. O primeiro treino veicular
# (work_dirs/lift3d_veicular) colapsava face, mãos e pernas --- interpupilar
# reconstruída de 3,3mm no próprio Drive&Act, contra 71,1mm deste.
#
# Mesa: o treinado com corte de quadro, medido nesse enquadramento (quadril de
# 495,8 para 72,0mm no S7 com a geometria da webcam). O veicular corrigido
# também reconstrói uma anatomia plausível na webcam (interpupilar 58,5mm,
# canela 313,0mm), mas foi especializado numa câmera e oito sujeitos, e não foi
# medido contra o v3 no enquadramento de mesa.
LIFT_CONFIG_BY_MOUNTING = {
    'mesa': 'configs/lift3d_dstformer_h3wb_robusto_v3.py',
    'retrovisor': 'configs/lift3d_veicular.py',
}
LIFT_CHECKPOINT_BY_MOUNTING = {
    'mesa': 'work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth',
    'retrovisor': ('work_dirs/lift3d_veicular_peso/'
                   'best_MPJPE_whole_epoch_5.pth'),
}

# Config e checkpoint das medições já publicadas, pelo mesmo motivo de
# `POSE_CHECKPOINT`.
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
# Mesa: 1,17m medidos com trena da lente ao rosto em 21/09/2026. A escala foi
# verificada contra referência física na gravação com tabuleiro, em que o
# solvePnP dá 1,11m: a distância interpupilar reconstruída fica a 1,7% da real.
# A pose encolhida da gravação de 12/09 era distância declarada errada.
# Retrovisor: 0,66m, mediana medida na referência 3D do Drive&Act.
DEFAULT_SUBJECT_DEPTH_M = {'mesa': 1.17, 'retrovisor': 0.664}

# Precisão da inferência do lifting, o estágio que domina o caminho completo.
# Medido (scripts/benchmark_lifting.py, v3, 300 janelas do S7): float32 27,91ms
# e 39,31mm; float16 10,72ms e 39,35mm; bfloat16 11,72ms e 39,38mm. float16 corta
# a latência 2,6 vezes por 0,04mm.
LIFT_PRECISIONS = ('float32', 'float16', 'bfloat16')
DEFAULT_LIFT_PRECISION = 'float16'

# De quantos em quantos quadros a janela do lifting é montada. Passo 3 reproduz
# ao vivo os 100ms medianos entre quadros do H3WB, e foi medido: piorou o tremor
# em todas as regiões (pernas 16,78 contra 10,63mm) e a coerência de osso.
DEFAULT_TEMPORAL_STRIDE = 1

# Taxa nominal que o One Euro assume constante. Um desvio de alguns hertz
# desloca o corte efetivo na mesma proporção, sem quebrar o filtro.
FILTER_RATE_HZ = 30.0


def optional_confidence(text: str) -> float | None:
    """Lê o teto de confiança, aceitando "nenhum" para desligá-lo.

    Existe porque o teto não é um número sempre presente: ele é contrato com o
    treino do checkpoint, e um checkpoint que não o viu precisa recebê-lo
    ausente, não zerado --- zero é uma confiança, ausência não é.
    """
    if text.strip().lower() in ('', 'nenhum', 'none'):
        return None
    return float(text)


def to_model_domain(frame: np.ndarray, keep_color: bool) -> np.ndarray:
    """Converte o frame para o domínio em que o modelo foi treinado.

    O modelo é treinado em COCO-WholeBody convertido para escala de cinza, como
    proxy do infravermelho. Alimentá-lo com RGB o colocaria fora do domínio de
    treino e mediria outra coisa que não o sistema proposto. Os três canais são
    mantidos porque o backbone pré-treinado espera essa forma de entrada.
    """
    import cv2

    if keep_color:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def config_without_flip_test(config_path: str, out_dir: Path) -> str:
    """Desliga o flip test, que dobra o custo da pose sem valor em operação.

    Ele executa o modelo também sobre a imagem espelhada e faz a média dos mapas
    de resposta. Isso compra precisão, que é o que a avaliação de AP mede, e
    custa metade da taxa de quadros: medidos 12,44 ms por pessoa sem ele contra
    cerca de 24,9 ms com ele. Em operação a métrica é latência.
    """
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    cfg.model.test_cfg = dict(cfg.model.get('test_cfg', {}))
    cfg.model.test_cfg['flip_test'] = False

    patched = Path(out_dir) / 'pose_config_ao_vivo.py'
    cfg.dump(patched)
    return str(patched)


def lift_dtype(precision: str):
    """O tipo do PyTorch para a precisão pedida; `None` mantém float32."""
    import torch
    return None if precision == 'float32' else getattr(torch, precision)


def build_lifter(settings):
    """Monta o lifting 3D com a geometria da câmera, se ela for conhecida.

    Args:
        settings: qualquer objeto com `lift_cfg`, `lift_ckpt`, `calibracao`,
            `distancia`, `device`, `teto_confianca`, `passo_temporal` e
            `precisao` --- os argumentos do painel, que os scripts de medição
            reproduzem para medir o mesmo sistema.

    Returns:
        (lifter, calibrado). `lifter` é `None` quando o painel 3D está desligado.
    """
    if not settings.lift_ckpt:
        print('Sem lifting: o painel 3D fica vazio.')
        return None, False

    print('Carregando lifting 3D...')
    from src.models.sequence_lifter import CameraView, SequenceLifter

    intrinsics = load_intrinsics(settings.calibracao)
    camera = None
    if intrinsics is not None:
        camera = CameraView(focal_length_px=intrinsics.fx,
                            principal_point=(intrinsics.cx, intrinsics.cy),
                            subject_depth_m=settings.distancia)
        print(f'  calibração: fx {intrinsics.fx:.1f}px, centro '
              f'({intrinsics.cx:.0f}, {intrinsics.cy:.0f}), ocupante a '
              f'{settings.distancia:.2f}m')
    else:
        print(f'  sem calibração em {settings.calibracao}; escala herdada do '
              f'H3WB, pose aproximada em tamanho')

    lifter = SequenceLifter(settings.lift_cfg, settings.lift_ckpt,
                            settings.device, camera=camera,
                            unobserved_confidence=settings.teto_confianca,
                            frame_stride=settings.passo_temporal,
                            inference_dtype=lift_dtype(settings.precisao))
    return lifter, camera is not None
