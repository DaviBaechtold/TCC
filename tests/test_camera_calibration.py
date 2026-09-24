#!/usr/bin/env python
"""Verifica que a calibração recupera intrínsecos conhecidos.

Uma calibração errada não falha: ela devolve números plausíveis que corrompem
tudo o que depende de escala. Foi assim que a escala herdada do H3WB ampliou a
pose em 3,8 vezes no domínio veicular sem que a métrica adotada acusasse.

O teste projeta um tabuleiro com intrínsecos escolhidos, acrescenta ruído da
ordem do que a detecção subpixel deixa, e confere que a calibração os recupera.

Executar:  python tests/test_camera_calibration.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.camera_calibration import DEFAULT_PATTERN, calibrate

TRUE_FX, TRUE_FY = 900.0, 900.0
TRUE_CX, TRUE_CY = 640.0, 360.0
TRUE_DISTORTION = np.array([-0.22, 0.05, 0.0, 0.0, 0.0])
IMAGE_SIZE = (1280, 720)
SQUARE_SIZE_M = 0.025

# A detecção subpixel erra nessa ordem; injetar menos tornaria o teste otimista.
DETECTION_NOISE_PX = 0.1

# Tolerâncias folgadas de propósito: o teste protege contra erro de fórmula ou
# de convenção — linhas trocadas por colunas, milímetros por metros — e não
# contra variação numérica de terceira casa.
MAX_FOCAL_ERROR = 0.02
MAX_CENTER_ERROR = 0.02
MAX_DISTORTION_ERROR = 0.02


def main():
    rng = np.random.default_rng(0)
    columns, rows = DEFAULT_PATTERN

    model = np.zeros((rows * columns, 3), np.float32)
    model[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    model *= SQUARE_SIZE_M

    matrix = np.array([[TRUE_FX, 0, TRUE_CX],
                       [0, TRUE_FY, TRUE_CY],
                       [0, 0, 1]], float)

    views = []
    for _ in range(20):
        # Inclinações variadas: vistas todas de frente deixam a distorção
        # indeterminada, e a calibração "funciona" com coeficientes sem sentido.
        rotation = rng.uniform(-0.5, 0.5, 3)
        translation = np.array([rng.uniform(-0.08, 0.08),
                                rng.uniform(-0.06, 0.06),
                                rng.uniform(0.35, 0.70)])
        projected, _ = cv2.projectPoints(model, rotation, translation,
                                         matrix, TRUE_DISTORTION)
        noisy = projected + rng.normal(scale=DETECTION_NOISE_PX,
                                       size=projected.shape)
        views.append(noisy.astype(np.float32))

    got = calibrate(views, IMAGE_SIZE, square_size_m=SQUARE_SIZE_M)

    for name, truth, estimated, tolerance in [
            ('fx', TRUE_FX, got.fx, MAX_FOCAL_ERROR),
            ('fy', TRUE_FY, got.fy, MAX_FOCAL_ERROR),
            ('cx', TRUE_CX, got.cx, MAX_CENTER_ERROR),
            ('cy', TRUE_CY, got.cy, MAX_CENTER_ERROR)]:
        erro = abs(estimated - truth) / truth
        assert erro < tolerance, f'{name}: {estimated:.1f} contra {truth:.1f}'
        print(f'  {name:3s} {estimated:8.1f}  erro {100 * erro:.2f}%  OK')

    for indice, nome in [(0, 'k1'), (1, 'k2')]:
        erro = abs(got.distortion[indice] - TRUE_DISTORTION[indice])
        assert erro < MAX_DISTORTION_ERROR, f'{nome}: {got.distortion[indice]}'
        print(f'  {nome:3s} {got.distortion[indice]:+8.4f}  erro {erro:.4f}  OK')

    assert got.reprojection_error < 1.0, got.reprojection_error
    print(f'  reprojeção {got.reprojection_error:.3f} px  OK')
    print('\na calibração recupera intrínsecos conhecidos')


if __name__ == '__main__':
    main()
