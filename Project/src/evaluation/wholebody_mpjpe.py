"""MPJPE decomposto por região anatômica, em milímetros.

Camada Model. Complementa a métrica `MPJPE` do MMPose, que reporta apenas o
agregado e nas unidades dos próprios dados.

Duas razões para existir. A primeira é de unidade: o H3WB armazena as
coordenadas em metros, de modo que a métrica nativa reporta, por exemplo, 0,088
onde a literatura reporta 88mm. Comparar sem converter produz erro de três
ordens de grandeza. A segunda é de granularidade: o benchmark do H3WB reporta
whole-body, corpo, face e mãos separadamente, e é essa decomposição que revela o
que domina o erro — nos métodos publicados, as mãos, com 125,3mm contra 66,5mm
da face no melhor deles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.registry import METRICS

METERS_TO_MILLIMETERS = 1000.0

# Fronteiras dos blocos no layout COCO-WholeBody, compartilhado pelo H3WB.
KEYPOINT_REGIONS = {
    'whole': slice(0, 133),
    'body': slice(0, 17),
    'feet': slice(17, 23),
    'face': slice(23, 91),
    'hands': slice(91, 133),
}


@METRICS.register_module()
class WholeBodyMPJPE(BaseMetric):
    """MPJPE por região, em milímetros.

    Args:
        mode: `'mpjpe'` sem alinhamento, `'p-mpjpe'` com alinhamento de
            Procrustes, que remove ambiguidade de escala e rotação e isola o
            erro de forma da pose.
    """

    ALIGNMENT_BY_MODE = {'mpjpe': 'none', 'p-mpjpe': 'procrustes'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 collect_device: str = 'cpu',
                 prefix: str | None = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if mode not in self.ALIGNMENT_BY_MODE:
            raise ValueError(
                f'Modo inválido: {mode}. '
                f'Use um de {sorted(self.ALIGNMENT_BY_MODE)}.')
        self.mode = mode

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            predicted = data_sample['pred_instances']['keypoints']
            if predicted.ndim == 4:
                predicted = np.squeeze(predicted, axis=0)

            ground_truth = data_sample['gt_instances']
            target = ground_truth['lifting_target']
            visible = ground_truth['lifting_target_visible'].astype(bool)

            self.results.append({
                'predicted': predicted,
                'target': target,
                'visible': visible.reshape(target.shape[0], -1),
            })

    def compute_metrics(self, results: list) -> dict[str, float]:
        predicted = np.concatenate([r['predicted'] for r in results])
        target = np.concatenate([r['target'] for r in results])
        visible = np.concatenate([r['visible'] for r in results])

        alignment = self.ALIGNMENT_BY_MODE[self.mode]
        error_name = self.mode.upper()
        metrics: dict[str, float] = {}

        for region_name, region in KEYPOINT_REGIONS.items():
            # Uma região sem nenhum keypoint anotado produziria média de um
            # conjunto vazio; reportar zero ali seria pior que omitir.
            region_visible = visible[:, region]
            if not region_visible.any():
                continue

            error_meters = keypoint_mpjpe(
                predicted[:, region], target[:, region], region_visible,
                alignment)
            metrics[f'{error_name}/{region_name}'] = (
                float(error_meters) * METERS_TO_MILLIMETERS)

        return metrics
