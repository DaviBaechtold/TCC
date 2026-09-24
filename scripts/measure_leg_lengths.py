#!/usr/bin/env python
"""Mede se fixar o comprimento dos ossos da perna piora a precisão.

Controller. A restrição de `src/models/leg_lengths.py` existe para estabilizar a
perna prevista ao vivo, onde não há referência. Antes de ligá-la, esta medição
responde à pergunta que o ao vivo não responde: ela afasta a perna da verdade?

O conjunto é o sujeito retido S7 do H3WB, que tem as pernas anotadas, nas três
condições do protocolo de corte de quadro (`src/evaluation/truncation_protocol.py`)
e com os mesmos parâmetros da medição publicada do v3. A âncora vem do braço
previsto, que é o que o painel tem; a variante com o braço verdadeiro separa o
erro da âncora do erro das proporções antropométricas.

    python scripts/measure_leg_lengths.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BATCH_SIZE = 16
LEGS = [13, 14, 15, 16]
FEET = list(range(17, 23))
MILLIMETERS = 1000.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lift-cfg', default='configs/lift3d_dstformer_h3wb_robusto_v3.py')
    p.add_argument('--lift-ckpt',
                   default='work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth')
    p.add_argument('--unobserved-confidence', type=float, default=0.3,
                   help='O teto da medição publicada do v3')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path,
                   default=Path('results/comprimento_perna_v3.json'))
    return p.parse_args()


def region_error_mm(predicted: np.ndarray, target: np.ndarray,
                    joints: list[int]) -> float:
    return float(np.linalg.norm(predicted[:, joints] - target[:, joints],
                                axis=-1).mean() * MILLIMETERS)


def main():
    args = parse_args()

    from src.models import torch_compat  # noqa: F401
    from src.data.estimator_noise import SimulatedEstimatorNoise
    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from src.data.h3wb_dataset import (last_frame_target,
                                       load_validation_windows, window_factor)
    from src.evaluation.truncation_protocol import (CONDITIONS, bone_geometry,
                                                    corrupt)
    from src.models.leg_lengths import (constrain_legs, leg_lengths,
                                        lengths_from_upper_arm,
                                        upper_arm_length)
    from src.models.sequence_lifter import H3WB_IMAGE_SIZE, SequenceLifter

    samples = load_validation_windows(args.lift_cfg)
    targets = np.stack([last_frame_target(s) for s in samples])
    factors = np.array([window_factor(s) for s in samples])
    inputs = np.stack([s['inputs'].numpy() for s in samples])
    print(f'{len(samples)} janelas do S7')

    lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device)
    noise = SimulatedEstimatorNoise(unobserved_confidence=args.unobserved_confidence)

    report = {'checkpoint': Path(args.lift_ckpt).name, 'janelas': len(samples),
              'entrada_2d': 'ground truth do H3WB, sujeito retido S7, com o '
                            'protocolo de corte de quadro',
              'quadro': 'causal, último da janela de 16',
              'proporcoes': 'razão perna/braço do ground truth do H3WB, '
                            'sujeitos S1, S5 e S6',
              'condicoes': {}}
    for condition in CONDITIONS:
        np.random.seed(args.seed)      # a mesma corrupção da medição do v3
        corrupted = np.stack([corrupt(w, condition, noise)[0] for w in inputs])
        predicted = np.concatenate([
            lifter.predict_windows(corrupted[i:i + BATCH_SIZE],
                                   (H3WB_IMAGE_SIZE, H3WB_IMAGE_SIZE),
                                   factors[i:i + BATCH_SIZE])
            for i in range(0, len(corrupted), BATCH_SIZE)])

        # Autoconsistência: cada osso na mediana do que o modelo previu para
        # esta pessoa. O S7 inteiro é um sujeito, e ao vivo é uma pessoa diante
        # da câmera; não impõe proporção externa, só tira a oscilação.
        coxa, canela = (np.median(x) for x in leg_lengths(predicted))
        variantes = {
            'bruto': predicted,
            'braco_previsto': constrain_legs(
                predicted, *lengths_from_upper_arm(upper_arm_length(predicted))),
            'braco_verdadeiro': constrain_legs(
                predicted, *lengths_from_upper_arm(upper_arm_length(targets))),
            'autoconsistente': constrain_legs(predicted, coxa, canela),
        }
        report['condicoes'][condition] = {
            nome: {'mpjpe_pernas_mm': round(region_error_mm(pose, targets, LEGS), 2),
                   'mpjpe_pes_mm': round(region_error_mm(pose, targets, FEET), 2),
                   'ossos': bone_geometry(pose, targets)}
            for nome, pose in variantes.items()}
        linha = report['condicoes'][condition]
        print(f'  {condition:10s} ' + '  '.join(
            f'{nome} pernas {v["mpjpe_pernas_mm"]:6.1f} pés {v["mpjpe_pes_mm"]:6.1f}'
            for nome, v in linha.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'gravado em {args.out}')


if __name__ == '__main__':
    main()
