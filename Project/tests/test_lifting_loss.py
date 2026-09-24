#!/usr/bin/env python
"""Verifica a perda ponderada do lifting contra as duas falhas da original.

1. Com peso 1 em todo ponto, ela devolve o mesmo valor da
   `MPJPEVelocityJointLoss` do MMPose --- trocar uma pela outra não muda treino
   algum onde todo ponto tem referência, como o do H3WB.
2. Um ponto com peso zero não influencia a perda, qualquer que seja o alvo dele.
   É a propriedade que faltava ao primeiro lifting veicular, cujos pontos sem
   referência tinham todos o mesmo alvo e foram treinados contra ele.
3. Funciona em janelas de 16 quadros, que é onde o caminho ponderado da original
   quebra (15 velocidades contra 16 pesos).

Executar:  python tests/test_lifting_loss.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOTE, QUADROS, PONTOS = 3, 16, 133
TOLERANCIA = 1e-4


def main():
    from mmpose.models.losses import MPJPEVelocityJointLoss

    from src.models.lifting_loss import WeightedMPJPEVelocityLoss

    torch.manual_seed(0)
    saida = torch.randn(LOTE, QUADROS, PONTOS, 3)
    alvo = torch.randn(LOTE, QUADROS, PONTOS, 3)
    perda = WeightedMPJPEVelocityLoss()

    # 1. Equivalência com a original quando todo ponto tem referência.
    original = MPJPEVelocityJointLoss()(saida, alvo)
    ponderada = perda(saida, alvo, torch.ones(LOTE, QUADROS, PONTOS, 1))
    assert abs(original.item() - ponderada.item()) < TOLERANCIA, (
        f'{ponderada.item():.6f} contra {original.item():.6f} da original: '
        'com peso 1 as duas precisam coincidir')
    print(f'  peso 1: {ponderada.item():.6f}, igual à original ({original.item():.6f})')

    # 2. Pontos sem referência não influenciam, qualquer que seja o alvo deles.
    peso = torch.ones(LOTE, QUADROS, PONTOS, 1)
    sem_referencia = slice(17, PONTOS)          # como no Drive&Act: só o corpo
    peso[:, :, sem_referencia] = 0.0
    base = perda(saida, alvo, peso)
    alvo_vazio = alvo.clone()
    alvo_vazio[:, :, sem_referencia] = 0.0      # todos no mesmo ponto
    alvo_absurdo = alvo.clone()
    alvo_absurdo[:, :, sem_referencia] = 1e3
    for nome, outro in (('vazio', alvo_vazio), ('absurdo', alvo_absurdo)):
        valor = perda(saida, outro, peso)
        assert abs(valor.item() - base.item()) < TOLERANCIA, (
            f'alvo {nome} nos pontos sem referência mudou a perda: '
            f'{valor.item():.6f} contra {base.item():.6f}')
    print('  pontos com peso zero não influenciam a perda, qualquer que seja o alvo')

    # E a predição desses pontos também não recebe gradiente.
    saida_grad = saida.clone().requires_grad_(True)
    perda(saida_grad, alvo, peso).backward()
    assert saida_grad.grad[:, :, sem_referencia].abs().max() == 0, (
        'pontos sem referência receberam gradiente')
    print('  e não recebem gradiente')

    print('\na perda respeita o peso e coincide com a original onde tudo tem referência')


if __name__ == '__main__':
    main()
