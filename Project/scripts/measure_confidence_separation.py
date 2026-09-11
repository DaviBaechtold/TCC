#!/usr/bin/env python
"""Mede quanto a resposta do estimador 2D separa o observado do invisível.

A adaptação ao domínio veicular, na primeira versão, deixou o modelo mais
confiante sobre o que ele não enxerga: a resposta nas juntas fora de quadro
subiu de 3,59 para 5,13, e a fração delas acima do limiar de 3,0 foi de 51%
para 92%. Este script é o que verifica se a supervisão negativa reverte isso.

O número que importa não é a resposta média em si, e sim a **separação** entre
as duas populações: é ela que determina se um limiar consegue distingui-las.

Exemplo:
    python scripts/measure_confidence_separation.py --ckpt <checkpoint>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCORE_THRESHOLD = 3.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--frames', type=Path,
                   default=Path('data/processed/driveact/val'))
    p.add_argument('--annotations', type=Path,
                   default=Path('data/processed/driveact/'
                                'driveact_midlevel.chunks_90.split_0.val.json'))
    p.add_argument('--max-frames', type=int, default=150)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path,
                   default=Path('results/separacao_confianca.json'))
    return p.parse_args()


def main():
    args = parse_args()

    import cv2

    from src.models import torch_compat  # noqa: F401
    from src.models.observability import (MIRROR_VIEW_ABSENT,
                                          MIRROR_VIEW_OBSERVABLE)
    from src.models.pose_pipeline import FullBodyPosePipeline

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)

    work_dir = Path('work_dirs/separacao')
    work_dir.mkdir(parents=True, exist_ok=True)
    pose = FullBodyPosePipeline(
        panel._config_without_flip_test(panel.POSE_CONFIG, work_dir),
        args.ckpt, args.device, detector=None)

    images = json.loads(args.annotations.read_text())['images'][:args.max_frames]
    observed, absent = [], []
    for image in images:
        frame = cv2.imread(str(args.frames / image['file_name']))
        if frame is None:
            continue
        result = pose(frame)
        if not result.num_people:
            continue
        observed.append(result.scores[0][list(MIRROR_VIEW_OBSERVABLE)])
        absent.append(result.scores[0][list(MIRROR_VIEW_ABSENT)])

    observed, absent = np.concatenate(observed), np.concatenate(absent)
    report = {
        'checkpoint': Path(args.ckpt).name,
        'resposta_observaveis': round(float(observed.mean()), 2),
        'resposta_ausentes': round(float(absent.mean()), 2),
        'separacao': round(float(observed.mean() / absent.mean()), 2),
        'ausentes_acima_do_limiar': round(
            float((absent >= SCORE_THRESHOLD).mean()), 3),
        'observaveis_acima_do_limiar': round(
            float((observed >= SCORE_THRESHOLD).mean()), 3),
        'quadros': len(images),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
