#!/usr/bin/env python
"""Prepara as anotações do H3WB para o formato do MMPose.

Controller: lê argumentos, chama `src/data/h3wb.py` e reporta o resultado.

Exemplo:
    python scripts/convert_h3wb.py \\
        --source data/raw/h3wb/reformatado \\
        --output data/processed/h3wb/h3wb_annotations.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.h3wb import convert


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True,
                        help='Diretório com train_data.npy e metadata.npy')
    parser.add_argument('--output', type=Path, required=True,
                        help='Arquivo .npz de saída')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = args.source / 'train_data.npy'
    metadata = args.source / 'metadata.npy'

    for path in (train_data, metadata):
        if not path.exists():
            raise SystemExit(f'Arquivo não encontrado: {path}')

    print(f'Lendo {args.source}...')
    summary = convert(train_data, metadata, args.output)

    size_mb = args.output.stat().st_size / 2**20
    print(f'\nsujeitos       {", ".join(summary.subjects)}')
    print(f'sequências     {summary.num_sequences}')
    print(f'caixas         {summary.num_boxes:,}')
    print(f'keypoints      {summary.num_keypoints}')
    print(f'\n{args.output}  ({size_mb:.0f} MB)')


if __name__ == '__main__':
    main()
