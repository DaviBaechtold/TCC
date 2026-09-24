#!/usr/bin/env python
"""Grava em arquivo um quadro do painel, sem abrir janela.

Controller. Gera as figuras do painel para os documentos a partir do próprio
`run_panel.main()`, e não de uma reconstrução: troca `imshow` e `waitKey` do
OpenCV por captura em arquivo e sai no quadro pedido. Assim a figura mostra o
sistema de operação --- estimador, lifting e filtro da montagem escolhida.

    python scripts/render_panel_frame.py --saida figura.png --quadro 400 -- \\
        --montagem retrovisor --source <video>.mp4 --calibracao <video>.calibration.json
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _QuadroGravado(Exception):
    """Interrompe o laço do painel quando o quadro pedido foi gravado."""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--saida', type=Path, required=True)
    parser.add_argument('--quadro', type=int, default=400)
    args, argumentos_do_painel = parser.parse_known_args()
    if argumentos_do_painel[:1] == ['--']:
        argumentos_do_painel = argumentos_do_painel[1:]

    renderizados = {'n': 0}

    def imshow(_janela, imagem):
        renderizados['n'] += 1
        if renderizados['n'] >= args.quadro:
            args.saida.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.saida), imagem)
            raise _QuadroGravado()

    cv2.imshow = imshow
    cv2.namedWindow = lambda *a, **k: None
    cv2.setMouseCallback = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0xFF

    spec = importlib.util.spec_from_file_location(
        'painel', Path(__file__).with_name('run_panel.py'))
    painel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(painel)
    sys.argv = ['run_panel.py'] + argumentos_do_painel
    try:
        painel.main()
    except _QuadroGravado:
        print(f'quadro {renderizados["n"]} gravado em {args.saida}')


if __name__ == '__main__':
    main()
