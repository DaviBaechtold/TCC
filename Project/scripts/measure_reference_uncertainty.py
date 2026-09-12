#!/usr/bin/env python
"""Estima a incerteza da referência 3D do Drive&Act por comprimento de osso.

A validação do Módulo 3 no domínio veicular compara contra a pose tridimensional
do Drive&Act, que **não** vem de captura com marcadores: ela é produto de
triangulação por OpenPose entre câmeras calibradas, e os autores do dataset não
caracterizam o erro dela. Sem saber esse erro, não se sabe se uma diferença de
13mm entre dois modelos é sinal ou ruído da régua.

O método não precisa de ground truth: **o comprimento de um osso é constante**.
A distância entre ombro e cotovelo da mesma pessoa não muda entre quadros, de
modo que toda variação medida nela é erro de reconstrução. Nenhum movimento real
pode produzi-la.

É um limite inferior da incerteza, não o erro absoluto: um viés sistemático que
encurte todos os ossos igualmente não apareceria aqui.

Exemplo:
    python scripts/measure_reference_uncertainty.py \\
        --poses ~/Downloads/extracted/openpose_3d
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ossos cujos extremos a vista de retrovisor observa, no layout COCO-WholeBody.
# Só entram pares em que ambas as pontas são observáveis: um osso com uma ponta
# extrapolada mediria a extrapolação, não a referência.
BONES = {
    'ombro a ombro': (5, 6),
    'braço esquerdo': (5, 7),
    'braço direito': (6, 8),
    'antebraço esquerdo': (7, 9),
    'antebraço direito': (8, 10),
    'quadril a quadril': (11, 12),
    'tronco esquerdo': (5, 11),
    'tronco direito': (6, 12),
}

# Abaixo disto a triangulação não produziu o ponto e o valor é preenchimento.
MIN_CONFIDENCE = 0.1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--max-sequences', type=int, default=8)
    p.add_argument('--out', type=Path,
                   default=Path('results/incerteza_referencia.json'))
    return p.parse_args()


def main():
    args = parse_args()

    from src.data.driveact import read_pose_csv

    arquivos = sorted(args.poses.glob('*/*.openpose.3d.csv'))[:args.max_sequences]
    if not arquivos:
        raise SystemExit(f'nenhum CSV de pose em {args.poses}')

    # Por sequência, e não agregado: cada sequência é uma pessoa, e agregar
    # misturaria diferença anatômica com erro de reconstrução.
    por_osso: dict[str, list[float]] = {nome: [] for nome in BONES}
    quadros = 0

    for arquivo in arquivos:
        comprimentos: dict[str, list[float]] = {nome: [] for nome in BONES}
        for frame in read_pose_csv(arquivo):
            quadros += 1
            for nome, (a, b) in BONES.items():
                if min(frame.confidence[a], frame.confidence[b]) < MIN_CONFIDENCE:
                    continue
                comprimentos[nome].append(
                    float(np.linalg.norm(frame.points_3d[a] - frame.points_3d[b])))

        for nome, valores in comprimentos.items():
            if len(valores) < 100:
                continue
            v = np.array(valores)
            # Desvio relativo à mediana da própria sequência: é a dispersão que
            # não pode vir de movimento, expressa em milímetros.
            por_osso[nome].append(float(np.std(v) * 1000))

    relatorio = {
        'sequencias': len(arquivos),
        'quadros': quadros,
        'metodo': 'desvio do comprimento de osso, que é fisicamente constante',
        'ossos': {},
    }
    print(f'{"osso":22s} {"desvio mediano":>15s} {"sequências":>11s}')
    for nome, desvios in por_osso.items():
        if not desvios:
            continue
        mediano = float(np.median(desvios))
        relatorio['ossos'][nome] = round(mediano, 2)
        print(f'{nome:22s} {mediano:13.1f}mm {len(desvios):11d}')

    if relatorio['ossos']:
        global_ = float(np.median(list(relatorio['ossos'].values())))
        relatorio['incerteza_mediana_mm'] = round(global_, 2)
        print(f'\nincerteza mediana da referência: {global_:.1f}mm')
        print('É limite inferior: um viés que encurte todos os ossos igualmente '
              'não apareceria neste teste.')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
