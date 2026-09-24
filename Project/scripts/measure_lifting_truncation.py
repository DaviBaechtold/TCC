#!/usr/bin/env python
"""Mede um checkpoint de lifting sob os cortes de quadro das duas montagens.

Controller: lê argumentos, monta o conjunto de validação do H3WB, aplica as
condições de `src/evaluation/truncation_protocol.py` e grava o relatório. Toda
regra de corrupção e de erro vive no Model.

    python scripts/measure_lifting_truncation.py \\
        --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v3.py \\
        --lift-ckpt work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth \\
        --tag v3

O sujeito retido é o S7, que nenhum dos treinos viu. As poses sentadas saem
separadas porque o ocupante de veículo e o usuário de mesa estão sentados,
enquanto o Human3.6M é majoritariamente em pé --- e é justamente a perna que
esses dois enquadramentos escondem.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# O H3WB normaliza contra uma imagem de 1000x1000.
IMAGE_SIZE = 1000

# Lote da inferência. Cabe com folga nos 8 GB e a janela é pequena; o limite
# aqui é a leitura do conjunto, não a GPU.
BATCH_SIZE = 16

SEATED_ACTIONS = ('Sitting', 'SittingDown')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lift-cfg', default='configs/lift3d_dstformer_h3wb_16frm.py')
    parser.add_argument('--lift-ckpt', required=True)
    parser.add_argument('--tag', required=True,
                        help='Nome do checkpoint no relatório')
    parser.add_argument('--unobserved-confidence', type=float, default=None,
                        help='Teto de confiança das juntas ocultas. Precisa '
                             'casar com o que o checkpoint viu no treino: o v3 '
                             'usa 0,3, o base e o v2 não usam teto')
    parser.add_argument('--colocacao', default='linha',
                        choices=('linha', 'medido'),
                        help='Mecanismo de colocação da junta cortada. Medir um '
                             'checkpoint sob o mecanismo do outro separa '
                             'generalização de aderência ao próprio treino')
    parser.add_argument('--max-windows', type=int, default=0,
                        help='0 usa todas as janelas do conjunto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def is_seated(sample: dict) -> bool:
    path = str(sample['data_samples'].metainfo['target_img_path'])
    return any(action in path for action in SEATED_ACTIONS)


def main():
    args = parse_args()

    from src.models import torch_compat  # noqa: F401
    from src.data.estimator_noise import SimulatedEstimatorNoise
    from src.evaluation.truncation_protocol import (CONDITIONS, bone_geometry,
                                                    corrupt, decompose_error)
    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from src.data.h3wb_dataset import (last_frame_target,
                                       load_validation_windows,
                                       window_factor)
    from src.models.sequence_lifter import SequenceLifter

    samples = load_validation_windows(args.lift_cfg, args.max_windows)
    targets = np.stack([last_frame_target(s) for s in samples])
    factors = np.array([window_factor(s) for s in samples])
    seated = np.array([is_seated(s) for s in samples])
    inputs = np.stack([s['inputs'].numpy() for s in samples])
    print(f'{len(samples)} janelas do S7, {int(seated.sum())} sentadas')

    lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device)
    noise = SimulatedEstimatorNoise(
        unobserved_confidence=args.unobserved_confidence,
        cut_placement=args.colocacao)

    report = {
        'checkpoint': Path(args.lift_ckpt).name,
        'config': Path(args.lift_cfg).name,
        'tag': args.tag,
        'janelas': len(samples),
        'janelas_sentadas': int(seated.sum()),
        'quadro': 'causal, último da janela de 16',
        'entrada_2d': 'ground truth do H3WB, sujeito retido S7',
        'teto_de_confianca': args.unobserved_confidence,
        'colocacao': args.colocacao,
        'nivel_do_corte_mesa': None,
        'condicoes': {},
    }

    for condition in CONDITIONS:
        np.random.seed(args.seed)      # mesma corrupção para todo checkpoint
        corrupted, hidden = [], []
        for window in inputs:
            janela, oculto = corrupt(window, condition, noise)
            corrupted.append(janela)
            hidden.append(oculto[-1])  # o quadro medido é o último
        corrupted = np.stack(corrupted)
        hidden = np.stack(hidden)

        predicted = np.concatenate([
            lifter.predict_windows(corrupted[start:start + BATCH_SIZE],
                                   (IMAGE_SIZE, IMAGE_SIZE),
                                   factors[start:start + BATCH_SIZE])
            for start in range(0, len(corrupted), BATCH_SIZE)
        ])

        resultado = decompose_error(predicted, targets, hidden)
        resultado['ossos'] = bone_geometry(predicted, targets)
        resultado['juntas_ocultas_por_janela'] = round(
            float(hidden.sum(axis=-1).mean()), 1)
        if seated.any():
            resultado['sentado'] = decompose_error(
                predicted[seated], targets[seated], hidden[seated])
        report['condicoes'][condition] = resultado

        print(f'  {condition:11s} mpjpe {resultado["mpjpe_mm"]:7.1f}  '
              f'visível {resultado["mpjpe_visivel_mm"]:6.1f}  '
              f'oculto {str(resultado["mpjpe_oculto_mm"] and round(resultado["mpjpe_oculto_mm"], 1)):>7}  '
              f'quadris {resultado["mpjpe_quadris_mm"]:6.1f}  '
              f'pernas {resultado["mpjpe_pernas_mm"]:6.1f}')

    from src.evaluation.truncation_protocol import MESA_CUT_LEVEL
    report['nivel_do_corte_mesa'] = MESA_CUT_LEVEL

    out = args.out or Path(f'results/truncamento_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
