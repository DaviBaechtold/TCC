#!/usr/bin/env python
"""Calibra a câmera do projeto com um tabuleiro de xadrez impresso.

Controller: abre a câmera, mostra o que está sendo detectado e grava o
resultado. Toda a regra de calibração vive em `src/data/camera_calibration.py`.

**Por que a demonstração precisa disto.** A escala que converte a saída do
Módulo 3 em metros vale `Z/fx`. Herdar o `fx` do H3WB em vez de medir o da
câmera própria ampliou a pose em 3,8 vezes no Drive&Act, e o erro absoluto foi a
1,4 metro sem que a métrica adotada acusasse. Ao vivo, o defeito seria o mesmo.

Uso, em duas etapas:

    # 1. Gera o tabuleiro em PDF com o quadrado do tamanho pedido e imprime.
    #    Escala 100%, sem "ajustar à página": o PDF traz uma régua de 100mm
    #    justamente para que o reescalonamento seja detectado antes de calibrar.
    python scripts/calibrate_camera.py --gerar-tabuleiro tabuleiro.pdf

    # 2. Calibra. O lado é o mesmo declarado na geração, salvo se a régua
    #    impressa acusar que a impressora reescalou.
    python scripts/calibrate_camera.py --lado-quadrado 0.025

Durante a captura: mova o tabuleiro pelo campo de visão, inclinando-o em
direções diferentes. Vistas todas de frente e no centro deixam a distorção
indeterminada, e a calibração "funciona" com coeficientes sem sentido.

Teclas: espaço captura a vista atual, c calibra com o que já foi capturado,
q sai sem gravar.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data.camera_calibration import (DEFAULT_PATTERN, MIN_VIEWS,
                                         calibrate, chessboard_image,
                                         chessboard_pdf, find_corners)

# Vistas quase idênticas não acrescentam informação e dão falsa confiança de
# cobertura. Meio segundo entre capturas é o suficiente para mover o tabuleiro.
MIN_INTERVAL_S = 0.5

JANELA = 'Calibracao — espaco captura, c calibra, q sai'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gerar-tabuleiro', type=Path, default=None,
                   help='Grava o padrão para impressão e sai')
    p.add_argument('--lado-quadrado', type=float, default=0.025,
                   help='Lado do quadrado impresso, em metros. Medir com régua: '
                        'ele fixa a escala do mundo')
    p.add_argument('--source', default='0')
    p.add_argument('--largura', type=int, default=1280)
    p.add_argument('--altura', type=int, default=720)
    p.add_argument('--out', type=Path,
                   default=Path('configs/camera/webcam.calibration.json'))
    return p.parse_args()


def main():
    args = parse_args()

    if args.gerar_tabuleiro:
        colunas, linhas = DEFAULT_PATTERN
        if args.gerar_tabuleiro.suffix.lower() == '.pdf':
            # PDF porque ele carrega dimensão física; o PNG não, e quem imprime
            # decide o tamanho sem que nada registre qual foi.
            chessboard_pdf(args.gerar_tabuleiro,
                           square_size_mm=args.lado_quadrado * 1000)
            print(f'Tabuleiro em {args.gerar_tabuleiro}: {colunas}x{linhas} '
                  f'cantos internos, quadrado de '
                  f'{args.lado_quadrado * 1000:.0f}mm, A4 paisagem.')
            print('Imprima em escala 100%, sem "ajustar à página". Confira a '
                  'régua de 100mm impressa no rodapé antes de calibrar.')
        else:
            cv2.imwrite(str(args.gerar_tabuleiro), chessboard_image())
            print(f'Tabuleiro em {args.gerar_tabuleiro}: {colunas}x{linhas} '
                  f'cantos internos, {colunas + 1}x{linhas + 1} quadrados. '
                  f'Sem escala física — prefira o PDF para imprimir.')
        print('Cole numa superfície rígida: papel ondulado curva o plano e '
              'contamina a distorção.')
        return

    import time

    fonte = int(args.source) if args.source.isdigit() else args.source
    captura = cv2.VideoCapture(fonte)
    # MJPG evita o teto de ~5 FPS que o YUYV impõe em 1280x720 nas webcams USB.
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)
    if not captura.isOpened():
        raise SystemExit(f'não foi possível abrir a câmera {args.source}')

    vistas = []
    tamanho = None
    ultima = 0.0
    print(f'Capture ao menos {MIN_VIEWS} vistas, variando posição e inclinação.')

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            tamanho = (frame.shape[1], frame.shape[0])
            cantos = find_corners(frame)

            tela = frame.copy()
            if cantos is not None:
                cv2.drawChessboardCorners(tela, DEFAULT_PATTERN, cantos, True)
            cor = (80, 220, 120) if cantos is not None else (80, 80, 230)
            cv2.putText(tela, f'vistas: {len(vistas)}/{MIN_VIEWS}', (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
            cv2.imshow(JANELA, tela)

            tecla = cv2.waitKey(1) & 0xFF
            agora = time.time()
            if tecla == ord(' ') and cantos is not None and \
                    agora - ultima > MIN_INTERVAL_S:
                vistas.append(cantos)
                ultima = agora
                print(f'  vista {len(vistas)} capturada')
            elif tecla == ord('c'):
                break
            elif tecla == ord('q'):
                print('saindo sem gravar')
                return
    finally:
        captura.release()
        cv2.destroyAllWindows()

    if len(vistas) < MIN_VIEWS:
        raise SystemExit(f'{len(vistas)} vistas capturadas; o mínimo é {MIN_VIEWS}')

    intrinsecos = calibrate(vistas, tamanho, square_size_m=args.lado_quadrado)
    intrinsecos.save(args.out)

    print(f'\nfx {intrinsecos.fx:.1f}  fy {intrinsecos.fy:.1f}')
    print(f'centro ({intrinsecos.cx:.1f}, {intrinsecos.cy:.1f})')
    print(f'distorção k1={intrinsecos.distortion[0]:+.4f} '
          f'k2={intrinsecos.distortion[1]:+.4f}')
    print(f'erro de reprojeção {intrinsecos.reprojection_error:.3f} px '
          f'em {intrinsecos.views} vistas')
    if intrinsecos.reprojection_error > 1.0:
        print('  Acima de 1 px: refaça variando mais a inclinação, ou confira '
              'se o tabuleiro está plano.')
    print(f'\ngravado em {args.out}')


if __name__ == '__main__':
    main()
