#!/usr/bin/env python
"""Mede se a janela temporal recupera um ponto oculto por alguns quadros (QP2).

Controller. O protocolo está em `src/evaluation/temporary_occlusion.py`: punho e
mão direitos ocultos nos últimos `k` quadros da janela, congelados na última
posição vista, contra o controle sem contexto temporal.

    python scripts/measure_temporary_occlusion.py \\
        --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v3.py \\
        --lift-ckpt work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth \\
        --tag v3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SIZE = 1000          # o H3WB normaliza por um quadro de 1000x1000


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lift-cfg', required=True)
    parser.add_argument('--lift-ckpt', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-windows', type=int, default=0)
    parser.add_argument('--confianca', default='teto', choices=('teto', 'mantida'),
                        help='teto: o ponto oculto cai ao teto de ausência do treino; '
                             'mantida: só a posição congela, o que isola o efeito '
                             'da confiança do efeito da posição')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Janelas por passada; menor quando a GPU está dividida com um treino')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.models import torch_compat  # noqa: F401
    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from src.data.h3wb_dataset import (last_frame_target,
                                       load_validation_windows, window_factor)
    from src.evaluation.temporary_occlusion import (DURATIONS, OCCLUDED_GROUP,
                                                    joint_error_mm,
                                                    occlude_last_frames,
                                                    without_context)
    from src.data.estimator_noise import UNOBSERVED_CONFIDENCE_CAP
    from src.models.sequence_lifter import SequenceLifter

    samples = load_validation_windows(args.lift_cfg, args.max_windows)
    targets = np.stack([last_frame_target(s) for s in samples])
    factors = np.array([window_factor(s) for s in samples], dtype=np.float32)
    inputs = np.stack([s['inputs'].numpy() for s in samples])
    print(f'{len(samples)} janelas do S7')

    lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device)

    def predict(windows: np.ndarray) -> np.ndarray:
        return np.concatenate([
            lifter.predict_windows(windows[i:i + args.batch_size],
                                   (IMAGE_SIZE, IMAGE_SIZE),
                                   factors[i:i + args.batch_size])
            for i in range(0, len(windows), args.batch_size)])

    resultados = {}
    for quadros in DURATIONS:
        teto = UNOBSERVED_CONFIDENCE_CAP if args.confianca == 'teto' else 1.0
        ocluidas = np.stack([occlude_last_frames(w, OCCLUDED_GROUP, quadros, teto)
                             for w in inputs])
        com_janela = predict(ocluidas)
        sem_janela = predict(np.stack([without_context(w) for w in ocluidas]))
        resultados[str(quadros)] = {
            'mao_com_janela_mm': round(joint_error_mm(com_janela, targets,
                                                      OCCLUDED_GROUP), 2),
            'mao_sem_contexto_mm': round(joint_error_mm(sem_janela, targets,
                                                        OCCLUDED_GROUP), 2),
        }
        r = resultados[str(quadros)]
        print(f'  oculta há {quadros:2d} quadros: com janela '
              f'{r["mao_com_janela_mm"]:6.1f}mm, sem contexto '
              f'{r["mao_sem_contexto_mm"]:6.1f}mm')

    report = {
        'checkpoint': Path(args.lift_ckpt).name,
        'tag': args.tag,
        'janelas': len(samples),
        'entrada_2d': 'ground truth do H3WB, sujeito retido S7',
        'quadro': 'causal, último da janela de 16',
        'grupo_oculto': 'punho e mão direitos (22 pontos)',
        'oclusao': 'congelado na última posição vista',
        'confianca_do_ponto_oculto': args.confianca,
        'por_duracao_da_oclusao': resultados,
    }
    out = args.out or Path(f'results/oclusao_temporaria_{args.tag}_{args.confianca}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
