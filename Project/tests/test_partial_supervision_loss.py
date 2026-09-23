#!/usr/bin/env python
"""Verifica que todo treino com supervisão parcial respeita o peso do alvo.

A `MPJPEVelocityJointLoss` do MMPose recebe o `lifting_target_weight` da cabeça
e o descarta, a menos que `use_target_weight` esteja ligado --- e o padrão é
desligado. Nada avisa: o treino roda, a perda cai, e o que a validação mede,
que são só as juntas com referência, melhora. Foi assim que a primeira versão do
lifting veicular aprendeu a colapsar num ponto os 121 pontos sem referência do
Drive&Act, e o defeito foi lido como "o modelo quebra fora da montagem".

Este teste percorre os configs de lifting, e todo aquele que mistura um
conjunto com alvo parcial precisa ligar o peso.

Executar:  python tests/test_partial_supervision_loss.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Conjuntos cuja referência cobre só parte dos 133 pontos.
PARTIAL_TARGET_DATASETS = ('DriveActLiftDataset',)


def dataset_types(dataset_cfg) -> list[str]:
    tipos = [dataset_cfg.get('type')]
    for filho in dataset_cfg.get('datasets', []):
        tipos += dataset_types(filho)
    return tipos


def main():
    from mmengine.config import Config

    verificados = 0
    for caminho in sorted(Path('configs').glob('lift3d_*.py')):
        cfg = Config.fromfile(str(caminho))
        tipos = dataset_types(cfg.train_dataloader.dataset)
        if not any(t in PARTIAL_TARGET_DATASETS for t in tipos):
            continue
        perda = cfg.model.head.loss
        assert perda.get('use_target_weight', False), (
            f'{caminho.name} treina com alvo parcial ({tipos}) e a perda '
            f'{perda["type"]} ignora o peso do alvo: os pontos sem referência '
            'seriam supervisionados contra um alvo vazio')
        verificados += 1
        print(f'  {caminho.name}: alvo parcial com peso respeitado')

    assert verificados, 'nenhum config com alvo parcial encontrado'
    print('\ntodo treino com supervisão parcial respeita o peso do alvo')


if __name__ == '__main__':
    main()
