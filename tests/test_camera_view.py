#!/usr/bin/env python
"""Confere a álgebra que leva o 2D ao vivo à geometria de treino do H3WB.

Só numpy, sem modelo e sem GPU: o que se verifica aqui é uma conta, e ela vale
ou não vale independentemente de qual checkpoint está carregado. Um teste que
precisasse do DSTFormer esconderia um erro de escala dentro do erro do modelo.

A propriedade que importa é **quantas unidades normalizadas vale um metro** na
entrada da rede. O DSTFormer não normaliza escala internamente, então esse
número é a diferença entre estar dentro e fora da distribuição de treino: no
H3WB vale 0,4466, e a webcam de mesa entrega 1,5454 sem a correção.

Executar:  python tests/test_camera_view.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sequence_lifter import (H3WB_FACTOR, H3WB_IMAGE_SIZE,
                                        H3WB_PRINCIPAL_POINT_PX, CameraView,
                                        to_training_geometry)
# A normalização do codec é privada porque ninguém fora do lifter deve chamá-la;
# aqui ela é o objeto do teste, e reimplementá-la faria o teste conferir a si
# mesmo em vez de conferir o código.
from src.models.sequence_lifter import _normalize

# Webcam de mesa do projeto, de configs/camera/webcam.calibration.json.
WEBCAM = CameraView(focal_length_px=959.3827033010448,
                    principal_point=(618.0590833768997, 342.6858415681451),
                    subject_depth_m=0.97)

# Unidades normalizadas por metro que o treino do H3WB viu. Sai do fator: o
# decodificador multiplica por `largura/2 · fator/1000`, de modo que um metro
# vale `2/fator` unidades.
H3WB_UNITS_PER_METRE = 2.0 / H3WB_FACTOR

TOLERANCE = 1e-6


def project(points_3d: np.ndarray, camera: CameraView) -> np.ndarray:
    """Projeção pinhole, sem distorção, na profundidade da própria pose."""
    focal = camera.focal_length_px
    optical_x, optical_y = camera.principal_point
    depth = points_3d[..., 2]
    return np.stack([focal * points_3d[..., 0] / depth + optical_x,
                     focal * points_3d[..., 1] / depth + optical_y], axis=-1)


def normalized(points_3d: np.ndarray, camera: CameraView) -> np.ndarray:
    """O que a rede recebe: projetado, mapeado e normalizado pelo codec."""
    mapped = to_training_geometry(project(points_3d, camera), camera)
    return _normalize(mapped, H3WB_IMAGE_SIZE, H3WB_IMAGE_SIZE)


def metre_ruler(camera: CameraView) -> np.ndarray:
    """Três pontos na profundidade do sujeito, a um metro um do outro."""
    depth = camera.subject_depth_m
    return np.array([[0.0, 0.0, depth], [1.0, 0.0, depth], [0.0, 1.0, depth]])


def test_units_per_metre_match_training():
    ruler = normalized(metre_ruler(WEBCAM), WEBCAM)
    horizontal = abs(ruler[1, 0] - ruler[0, 0])
    vertical = abs(ruler[2, 1] - ruler[0, 1])

    print(f'  unidades/metro: {horizontal:.6f} em x, {vertical:.6f} em y '
          f'(H3WB: {H3WB_UNITS_PER_METRE:.6f})')
    assert abs(horizontal - H3WB_UNITS_PER_METRE) < TOLERANCE
    assert abs(vertical - H3WB_UNITS_PER_METRE) < TOLERANCE


def test_principal_point_maps_to_virtual_centre():
    """O eixo óptico da câmera real vai para o da virtual, não para o do quadro.

    O ponto principal do H3WB não coincide com o centro do quadro de 1000x1000,
    e é essa pequena assimetria que o treino viu.
    """
    optical_axis = np.array([[*WEBCAM.principal_point]])
    mapped = to_training_geometry(optical_axis, WEBCAM)

    print(f'  eixo óptico mapeado: {np.round(mapped[0], 4)} '
          f'(esperado {H3WB_PRINCIPAL_POINT_PX})')
    assert np.allclose(mapped[0], H3WB_PRINCIPAL_POINT_PX, atol=1e-3)


def test_independent_of_live_frame_size():
    """A mesma câmera em quadros diferentes entrega a mesma escala.

    O caminho antigo normalizava pela largura do quadro, de modo que trocar a
    resolução da webcam mudava silenciosamente a escala vista pela rede.
    """
    ruler = metre_ruler(WEBCAM)
    reference = normalized(ruler, WEBCAM)

    # O mapeamento não recebe o tamanho do quadro; o que muda com a resolução é
    # a calibração, e uma câmera com o dobro da largura tem o dobro de fx e de
    # ponto principal para o mesmo campo de visão.
    doubled = CameraView(focal_length_px=WEBCAM.focal_length_px * 2,
                         principal_point=tuple(
                             2 * value for value in WEBCAM.principal_point),
                         subject_depth_m=WEBCAM.subject_depth_m)
    print(f'  x normalizado do ponto a um metro: '
          f'{reference[1, 0]:.6f} em 1280px, '
          f'{normalized(ruler, doubled)[1, 0]:.6f} em 2560px')
    assert np.allclose(reference, normalized(ruler, doubled), atol=1e-5)


def test_synthetic_pose_round_trips_in_metres():
    """Uma pose sintética volta com os comprimentos certos depois do decode.

    Reproduz a decodificação do codec — `(unidades + trans)·largura/2 · fator`,
    em milímetros — sobre a pose normalizada, e confere que os ossos medem o que
    mediam em metros antes de projetar.
    """
    depth = WEBCAM.subject_depth_m
    # Ombros a 0,38m um do outro, quadril 0,50m abaixo do ombro esquerdo.
    pose = np.array([[-0.19, 0.0, depth],
                     [0.19, 0.0, depth],
                     [-0.19, 0.50, depth]])
    decoded = normalized(pose, WEBCAM) * (H3WB_IMAGE_SIZE / 2.0) * H3WB_FACTOR
    decoded /= 1000.0

    shoulders = abs(decoded[1, 0] - decoded[0, 0])
    torso = abs(decoded[2, 1] - decoded[0, 1])
    print(f'  ombros: {shoulders:.4f}m (0,3800)   tronco: {torso:.4f}m (0,5000)')
    assert abs(shoulders - 0.38) < 1e-4
    assert abs(torso - 0.50) < 1e-4


def test_uncorrected_scale_is_out_of_distribution():
    """Registra o tamanho do problema que a correção resolve.

    Sem o mapeamento, a normalização pela largura do quadro entrega 3,46 vezes
    a escala de treino. O número está aqui para que uma mudança de calibração
    que o altere apareça como falha, e não como pose estranha no painel.
    """
    frame_width = 1280
    uncorrected = 2.0 / frame_width * WEBCAM.pixels_per_metre
    ratio = uncorrected / H3WB_UNITS_PER_METRE

    print(f'  sem correção: {uncorrected:.4f} unidades/m, '
          f'{ratio:.2f}x a escala de treino')
    assert abs(ratio - 3.46) < 0.01


def main():
    for test in (test_units_per_metre_match_training,
                 test_principal_point_maps_to_virtual_centre,
                 test_independent_of_live_frame_size,
                 test_synthetic_pose_round_trips_in_metres,
                 test_uncorrected_scale_is_out_of_distribution):
        print(f'{test.__name__}:')
        test()

    print('\ngeometria virtual do H3WB confere')


if __name__ == '__main__':
    main()
