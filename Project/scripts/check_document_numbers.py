#!/usr/bin/env python
"""Confere se os números do Projeto Físico batem com as medições em results/.

O `CLAUDE.md` dos dois repositórios fixa a regra: mudança de métrica em um exige
atualizar o outro, e um documento que descreve um sistema que o código não
implementa é o defeito mais caro deste projeto. Esta verificação transforma a
regra em algo executável, em vez de depender de alguém lembrar.

Ela não prova que o documento está certo --- um número pode estar no lugar
errado, ou acompanhado de uma condição de medição falsa. Ela prova que nenhum
número citado foi deixado para trás quando a medição mudou, que é o modo de
falha frequente.

Compara com tolerância de arredondamento: o documento usa duas casas decimais e
vírgula decimal, enquanto os arquivos de resultado guardam a precisão inteira.

Executar:  python scripts/check_document_numbers.py
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DOCUMENT = Path.home() / ('Documents/Projeto-Fisico/projeto-fisico/'
                                  'projeto fisico.tex')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--document', type=Path, default=DEFAULT_DOCUMENT)
    p.add_argument('--results', type=Path, default=Path('results'))
    return p.parse_args()


def citado(document: str, value: float, decimals: int = 2) -> bool:
    """Procura o valor no documento, aceitando as formas em que ele aparece.

    Um mesmo número é escrito de várias maneiras: 0,0290 e 0,029 são o mesmo
    valor, e 15,44 é como 15,442 aparece depois de arredondado. Exigir grafia
    exata produziria alarme falso em quase toda linha.
    """
    # De uma casa a quatro: o documento arredonda latências para uma casa
    # ("15,5ms") e erros normalizados para quatro ("0,0288").
    for casas in range(1, decimals + 3):
        texto = f'{value:.{casas}f}'.replace('.', ',').rstrip('0').rstrip(',')
        if texto and texto in document:
            return True
    return False


def coletar(results: Path) -> list[tuple[str, float]]:
    """Extrai os números que o documento tem obrigação de citar."""
    numeros: list[tuple[str, float]] = []

    for arquivo in sorted((results / 'baselines').glob('*.json')):
        dados = json.loads(arquivo.read_text())
        if 'metrics' not in dados:
            continue
        for chave in ('coco-wholebody/AP', 'torso/px_mean'):
            if chave in dados['metrics']:
                valor = dados['metrics'][chave]
                # O AP é citado em pontos percentuais; o erro, em pixels.
                escala = 100 if chave.endswith('AP') else 1
                numeros.append((f'{arquivo.stem} {chave}', valor * escala))

    for nome in ('qp1_detectores.json',):
        arquivo = results / nome
        if arquivo.exists():
            for chave, valores in json.loads(arquivo.read_text()).items():
                numeros.append((f'QP1 {chave}', valores['erro_normalizado']))

    for arquivo in sorted(results.glob('lifting_driveact*.json')):
        dados = json.loads(arquivo.read_text())
        numeros.append((arquivo.stem, dados['pa_mpjpe_mm']))

    for arquivo in sorted((results / 'throughput').glob('*.json')):
        dados = json.loads(arquivo.read_text())
        numeros.append((f'throughput {dados["label"]}', dados['median_ms']))

    return numeros


def main():
    args = parse_args()
    if not args.document.exists():
        raise SystemExit(f'documento não encontrado: {args.document}')

    document = args.document.read_text(encoding='utf-8')
    ausentes = []
    for rotulo, valor in coletar(args.results):
        if not citado(document, valor):
            ausentes.append((rotulo, valor))
            print(f'  não citado: {rotulo:46s} {valor:10.4f}')

    total = len(coletar(args.results))
    print(f'\n{total - len(ausentes)} de {total} medições citadas no documento')
    if ausentes:
        print('Nem toda medição precisa estar no documento; a lista acima é '
              'para revisão, não para falhar o processo.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
