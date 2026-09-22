#!/usr/bin/env python
"""Decompõe o erro das mãos do lifting no sujeito retido do H3WB.

Controller. As mãos medem 77mm contra 40 do corpo e são a única meta numérica
ainda aberta (< 60mm). Antes de treinar qualquer coisa contra elas, esta medição
diz **onde** o erro está: no punho, que é erro do braço, na orientação da mão ou
na forma dela.

    python scripts/measure_hand_error.py --lift-cfg <cfg> --lift-ckpt <ckpt> --tag v3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SIZE = 1000
BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lift-cfg', default='configs/lift3d_dstformer_h3wb_robusto_v3.py')
    parser.add_argument('--lift-ckpt', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-windows', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.models import torch_compat  # noqa: F401
    from src.evaluation.hand_error import decompose
    from src.models.sequence_lifter import SequenceLifter

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'truncation', Path(__file__).with_name('measure_lifting_truncation.py'))
    protocolo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocolo)

    amostras = protocolo.load_windows(args.lift_cfg, args.max_windows)
    alvos = np.stack([protocolo.target_of(s) for s in amostras])
    fatores = np.array([float(np.asarray(
        s['data_samples'].metainfo['factor']).ravel()[-1]) for s in amostras])
    entradas = np.stack([s['inputs'].numpy() for s in amostras])

    lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device)
    preditos = np.concatenate([
        lifter.predict_windows(entradas[i:i + BATCH_SIZE],
                               (IMAGE_SIZE, IMAGE_SIZE),
                               fatores[i:i + BATCH_SIZE])
        for i in range(0, len(entradas), BATCH_SIZE)])

    relatorio = {
        'tag': args.tag,
        'checkpoint': Path(args.lift_ckpt).name,
        'janelas': len(amostras),
        'entrada_2d': 'ground truth do H3WB, sujeito retido S7, entrada íntegra',
        'quadro': 'causal, último da janela de 16',
        **decompose(preditos, alvos),
    }

    out = args.out or Path(f'results/erro_maos_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    for lado in ('esquerda', 'direita'):
        print(f'  mão {lado:9s} absoluto {relatorio[f"{lado}_absoluto_mm"]:6.1f}  '
              f'no punho {relatorio[f"{lado}_no_punho_mm"]:6.1f}  '
              f'forma {relatorio[f"{lado}_forma_mm"]:6.1f}  '
              f'punho {relatorio[f"{lado}_punho_mm"]:6.1f}  '
              f'tamanho {relatorio[f"{lado}_tamanho_predito_mm"]:5.1f} contra '
              f'{relatorio[f"{lado}_tamanho_verdadeiro_mm"]:5.1f}')
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
