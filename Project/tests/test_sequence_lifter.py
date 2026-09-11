#!/usr/bin/env python
"""Verifica que o lifting ao vivo reproduz o resultado da validação.

O caminho de inferência ao vivo é diferente do da validação: ele monta a janela
temporal quadro a quadro, em vez de recebê-la pronta do dataset. Se divergirem, o
painel exibe uma pose plausível porém errada, e nada denuncia.

A verificação alimenta o lifter com janelas reais do H3WB, desnormalizadas para
pixels, e confere que o erro fica na faixa da validação oficial.

Executar:  python tests/test_sequence_lifter.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = 'configs/lift3d_dstformer_h3wb_16frm.py'
CHECKPOINT = 'work_dirs/lift3d_dstformer_h3wb/best_MPJPE_whole_epoch_30.pth'

# A validação oficial mede 38,96mm sobre todas as posições da janela. O caminho
# ao vivo devolve a última, que custa 3,51mm a mais por não ter contexto futuro.
# A margem cobre a variação de amostragem de um subconjunto pequeno.
MAX_MPJPE_MM = 50.0
WINDOWS_TO_CHECK = 20

# O H3WB normaliza por uma imagem de 1000x1000.
IMAGE_SIZE = 1000


def main():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    from src.models import torch_compat  # noqa: F401

    init_default_scope('mmpose')
    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from mmpose.registry import DATASETS

    from src.models.sequence_lifter import SequenceLifter

    cfg = Config.fromfile(CONFIG)
    dataset_cfg = cfg.val_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = cfg.val_pipeline
    dataset = DATASETS.build(dataset_cfg)

    lifter = SequenceLifter(CONFIG, CHECKPOINT, 'cuda:0')

    errors = []
    step = max(1, len(dataset) // WINDOWS_TO_CHECK)
    for index in range(0, len(dataset), step):
        sample = dataset[index]
        lifter.reset()

        # O fator vem da geometria da câmera. Aqui ele é conhecido; ao vivo, sem
        # calibração, não é — e a diferença custa 29mm de MPJPE.
        lifter._factor = float(
            np.asarray(sample['data_samples'].metainfo['factor']).ravel()[-1])

        normalized = sample['inputs'].numpy()
        pixels = (normalized[..., :2] + np.array([1.0, 1.0])) * IMAGE_SIZE / 2.0
        for frame in range(normalized.shape[0]):
            predicted = lifter(pixels[frame], normalized[frame, :, 2],
                               (IMAGE_SIZE, IMAGE_SIZE))

        target = np.squeeze(
            np.asarray(sample['data_samples'].gt_instances.lifting_target))
        target = target[-1] if target.ndim == 3 else target
        target = target - target[:1]
        errors.append(np.linalg.norm(predicted - target, axis=-1).mean() * 1000)

    mean_error = float(np.mean(errors))
    print(f'  MPJPE do caminho ao vivo: {mean_error:.1f} mm '
          f'em {len(errors)} janelas')
    assert mean_error < MAX_MPJPE_MM, (
        f'{mean_error:.1f} mm acima do teto de {MAX_MPJPE_MM} mm: o caminho ao '
        'vivo divergiu do da validação')
    print('\ncaminho de inferência ao vivo reproduz a validação')


if __name__ == '__main__':
    main()
