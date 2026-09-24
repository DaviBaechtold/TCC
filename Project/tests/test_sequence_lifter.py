#!/usr/bin/env python
"""Verifica que o lifting ao vivo reproduz o resultado da validação.

O caminho de inferência ao vivo é diferente do da validação: ele monta a janela
temporal quadro a quadro, em vez de recebê-la pronta do dataset. Se divergirem, o
painel exibe uma pose plausível porém errada, e nada denuncia.

A verificação alimenta o lifter com janelas reais do H3WB, desnormalizadas para
pixels, e confere que o erro fica na faixa da validação oficial. Depois confere
que `predict_windows`, o caminho em lote usado nas medições, devolve a mesma
pose que a chamada quadro a quadro — as duas rotas precisam ser a mesma conta,
senão um número medido em lote não descreve o que o painel mostra.

Este teste roda com `camera=None`, de propósito: aqui a entrada **já está** na
geometria do H3WB, e mapeá-la para uma câmera virtual seria mapear o H3WB nele
mesmo. A álgebra do caminho com câmera é conferida em tests/test_camera_view.py.

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

# Divergência tolerada entre o lote e o quadro a quadro. Não é zero porque o
# lote atravessa o mesmo tensor em outra ordem de redução; 1mm é duas ordens de
# grandeza abaixo do erro do próprio modelo.
MAX_BATCH_GAP_M = 1e-3
WINDOWS_IN_BATCH = 4

# O H3WB normaliza por uma imagem de 1000x1000.
IMAGE_SIZE = 1000


def replay(lifter, normalized, factor):
    """Alimenta o lifter quadro a quadro, como o painel faz, e devolve a última.

    O fator vem da geometria da câmera. Aqui ele é conhecido; ao vivo, sem
    calibração, não é — e a diferença custa 29mm de MPJPE.
    """
    lifter.reset()
    lifter._factor = factor
    pixels = (normalized[..., :2] + np.array([1.0, 1.0])) * IMAGE_SIZE / 2.0
    for frame in range(normalized.shape[0]):
        predicted = lifter(pixels[frame], normalized[frame, :, 2],
                           (IMAGE_SIZE, IMAGE_SIZE))
    return predicted


def main():
    from src.models import torch_compat  # noqa: F401

    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from src.data.h3wb_dataset import (last_frame_target,
                                       load_validation_windows,
                                       window_factor)

    samples = load_validation_windows(CONFIG, WINDOWS_TO_CHECK)

    from src.models.sequence_lifter import SequenceLifter

    # O H3WB já traz visibilidade em [0, 1]; a escala de 1,0 evita dividir duas
    # vezes. O painel, que recebe resposta do SimCC, usa o padrão.
    lifter = SequenceLifter(CONFIG, CHECKPOINT, 'cuda:0', response_scale=1.0)

    errors = []
    predictions = []
    for sample in samples:
        predicted = replay(lifter, sample['inputs'].numpy(), window_factor(sample))
        predictions.append(predicted)
        errors.append(
            np.linalg.norm(predicted - last_frame_target(sample), axis=-1).mean() * 1000)

    mean_error = float(np.mean(errors))
    print(f'  MPJPE do caminho ao vivo: {mean_error:.1f} mm '
          f'em {len(errors)} janelas')
    assert mean_error < MAX_MPJPE_MM, (
        f'{mean_error:.1f} mm acima do teto de {MAX_MPJPE_MM} mm: o caminho ao '
        'vivo divergiu do da validação')

    batch = samples[:WINDOWS_IN_BATCH]
    in_batch = lifter.predict_windows(
        np.stack([sample['inputs'].numpy() for sample in batch]),
        (IMAGE_SIZE, IMAGE_SIZE),
        np.array([window_factor(sample) for sample in batch], dtype=np.float32))
    gap = np.abs(in_batch - np.stack(predictions[:WINDOWS_IN_BATCH])).max()
    print(f'  maior diferença entre lote e quadro a quadro: {gap * 1000:.4f} mm')
    assert gap < MAX_BATCH_GAP_M, (
        f'{gap * 1000:.2f} mm entre `predict_windows` e a chamada quadro a '
        'quadro: as duas rotas deixaram de ser a mesma conta')

    print('\ncaminho de inferência ao vivo reproduz a validação')


if __name__ == '__main__':
    main()
