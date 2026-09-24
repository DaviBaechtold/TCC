#!/usr/bin/env python
"""Filtra um COCO-WholeBody pelas instâncias que anotam face ou mãos.

Controller. Existe para o treino de ensaio: só 29,4% das instâncias do conjunto
de treino anotam face ou mãos, e uma instância sem anotação ali não produz
gradiente algum para a região que o ensaio existe para preservar. Misturar o
conjunto inteiro diluiria o sinal por três.

    python scripts/filter_wholebody_annotations.py --exigir qualquer

As imagens sem anotação remanescente saem junto, para que o dataset não carregue
índice de arquivo que nunca será lido.
"""

import argparse
import json
from pathlib import Path

ENTRADA = Path('data/processed/grayscale/annotations/coco_wholebody_train_v1.0.json')

CRITERIOS = {
    'face': lambda a: bool(a.get('face_valid')),
    'maos': lambda a: bool(a.get('lefthand_valid') or a.get('righthand_valid')),
    'qualquer': lambda a: bool(a.get('face_valid') or a.get('lefthand_valid')
                              or a.get('righthand_valid')),
    'ambos': lambda a: bool(a.get('face_valid')
                            and (a.get('lefthand_valid')
                                 or a.get('righthand_valid'))),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entrada', type=Path, default=ENTRADA)
    parser.add_argument('--saida', type=Path, default=None)
    parser.add_argument('--exigir', choices=sorted(CRITERIOS),
                        default='qualquer')
    return parser.parse_args()


def main():
    args = parse_args()
    dados = json.loads(args.entrada.read_text())
    mantem = CRITERIOS[args.exigir]

    anotacoes = [a for a in dados['annotations'] if mantem(a)]
    com_anotacao = {a['image_id'] for a in anotacoes}
    imagens = [i for i in dados['images'] if i['id'] in com_anotacao]

    dados['annotations'] = anotacoes
    dados['images'] = imagens

    saida = args.saida or args.entrada.with_name(
        f'{args.entrada.stem}_{args.exigir}.json')
    saida.write_text(json.dumps(dados))
    print(f'{len(anotacoes)} instâncias em {len(imagens)} imagens '
          f'(de {len(json.loads(args.entrada.read_text())["annotations"])})')
    print(f'gravado em {saida}')


if __name__ == '__main__':
    main()
