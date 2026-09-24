"""Onde vive o erro das mãos: no punho, na orientação ou na forma.

Camada Model. As mãos dominam o erro do Módulo 3 --- 77mm contra 40 do corpo ---
e são a única meta numérica ainda aberta do projeto. O número agregado não diz
o que corrigir: um erro de 77mm pode ser a mão inteira no lugar errado, com
forma perfeita, ou a mão no lugar certo com os dedos embaralhados. As duas
exigem correções opostas.

A decomposição separa três contribuições, cada uma removendo uma grandeza a
mais:

    absoluto     a mão como ela sai, ancorada na raiz da pose
    no punho     subtraindo o punho de cada lado, o que remove o erro de
                 posição e deixa orientação mais forma
    forma        alinhando a mão predita à verdadeira por Procrustes, o que
                 remove também orientação e escala

O que sobra em `forma` é o que uma perda específica de mão poderia atacar; o que
some entre `absoluto` e `no punho` é erro do braço, e se corrige a montante.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.pose_alignment import procrustes_align

MILLIMETERS = 1000.0

# Blocos do COCO-WholeBody: o punho é a junta corporal que ancora cada mão, e o
# primeiro keypoint de cada bloco de mão é a raiz dela, no mesmo lugar.
HANDS = {
    'esquerda': {'punho': 9, 'bloco': list(range(91, 112))},
    'direita': {'punho': 10, 'bloco': list(range(112, 133))},
}


def decompose(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Erro das mãos em milímetros, decomposto por grandeza removida.

    Args:
        predicted, target: [N, 133, 3] em metros, ancorados na raiz da pose.
    """
    relatorio: dict[str, float] = {}
    for lado, indices in HANDS.items():
        punho, bloco = indices['punho'], indices['bloco']

        absoluto = np.linalg.norm(
            predicted[:, bloco] - target[:, bloco], axis=-1)
        no_punho = np.linalg.norm(
            (predicted[:, bloco] - predicted[:, punho, None])
            - (target[:, bloco] - target[:, punho, None]), axis=-1)
        forma = np.stack([
            np.linalg.norm(procrustes_align(p, t) - t, axis=-1)
            for p, t in zip(predicted[:, bloco], target[:, bloco])
        ])

        relatorio[f'{lado}_absoluto_mm'] = float(absoluto.mean() * MILLIMETERS)
        relatorio[f'{lado}_no_punho_mm'] = float(no_punho.mean() * MILLIMETERS)
        relatorio[f'{lado}_forma_mm'] = float(forma.mean() * MILLIMETERS)
        relatorio[f'{lado}_punho_mm'] = float(np.linalg.norm(
            predicted[:, punho] - target[:, punho], axis=-1).mean() * MILLIMETERS)

        # Tamanho da mão predita contra a verdadeira: uma mão sistematicamente
        # menor é erro de escala, que nenhuma das três colunas acima isola.
        def envergadura(poses):
            mao = poses[:, bloco]
            return np.linalg.norm(
                mao.max(axis=1) - mao.min(axis=1), axis=-1).mean()

        relatorio[f'{lado}_tamanho_predito_mm'] = float(
            envergadura(predicted) * MILLIMETERS)
        relatorio[f'{lado}_tamanho_verdadeiro_mm'] = float(
            envergadura(target) * MILLIMETERS)
    return relatorio
