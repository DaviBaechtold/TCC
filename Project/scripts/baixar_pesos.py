#!/usr/bin/env python
"""Baixa os pesos que o sistema precisa, e confere cada um pelo SHA-256.

Controller. Os pesos não estão no git --- somam 1,4 GB --- e sem eles nada roda.
Há dois tipos:

- **Pesos de terceiros**, baixados dos endereços oficiais: o detector
  YOLO26n-pose e, só para treinar ou usar o detector antigo, o RTMW-x original e
  o RTMDet-nano.
- **Pesos treinados neste projeto**: os dois estimadores 2D (mesa e retrovisor)
  e os dois liftings 3D. Não são públicos: dois deles derivam do Drive&Act e do
  H3WB, de licença acadêmica. Ficam numa pasta do Google Drive compartilhada
  pelo autor, e entram por `--origem` com a pasta baixada.

Cada arquivo vai para o caminho que `src/models/operating_config.py` e
`src/models/detector_config.py` esperam; nada precisa ser configurado depois.

    python scripts/baixar_pesos.py --origem ~/Downloads/pesos   # para operar
    python scripts/baixar_pesos.py --origem ~/Downloads/pesos --todos
    python scripts/baixar_pesos.py --empacotar ~/TCC_pesos-v1   # para o autor
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import RTMDET_CHECKPOINT, YOLO_CHECKPOINT
from src.models.operating_config import (LIFT_CHECKPOINT_BY_MOUNTING,
                                         POSE_CHECKPOINT_BY_MOUNTING)

RTMW_X_OFFICIAL = ('checkpoints/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-'
                   'f840f204_20231122.pth')


@dataclass(frozen=True)
class Weight:
    """Um arquivo de pesos: onde ele fica, de onde vem e como conferi-lo."""

    destination: str
    sha256: str
    url: str | None = None        # None: treinado neste projeto, vem de --origem
    published_name: str | None = None
    training_only: bool = False


WEIGHTS = (
    Weight(POSE_CHECKPOINT_BY_MOUNTING['mesa'],
           '8a7effcf4cf0df2bea2410b372ec46c1ca88ae2e8811903b8fe4f0c6be72a9f0',
           published_name='pose_mesa_rtmw_x_gray_lora.pth'),
    Weight(POSE_CHECKPOINT_BY_MOUNTING['retrovisor'],
           '76e21b30d822c19c4f5edc990aae83eb75abb14b78dd7d2080d2b2741e0a304d',
           published_name='pose_retrovisor_rtmw_x_ensaio.pth'),
    Weight(LIFT_CHECKPOINT_BY_MOUNTING['mesa'],
           '00336590e78438bd372c4b53711dfa275da0d40a96ab4e43720c548abd31c477',
           published_name='lifting_mesa_v3.pth'),
    Weight(LIFT_CHECKPOINT_BY_MOUNTING['retrovisor'],
           'e1fecf6c79ac1ebdf7bc92220c5af430b9d176092596c8c6c123848353bd3463',
           published_name='lifting_retrovisor_veicular.pth'),
    Weight(YOLO_CHECKPOINT,
           'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9',
           url=('https://github.com/ultralytics/assets/releases/download/'
                'v8.4.0/yolo26n-pose.pt')),
    Weight(RTMW_X_OFFICIAL,
           'f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95',
           url=('https://download.openmmlab.com/mmpose/v1/projects/rtmw/'
                'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_'
                '20231122.pth'),
           training_only=True),
    Weight(RTMDET_CHECKPOINT,
           '0e2da635c75e25dc88af08d01eb34bfe9cac06a7841cba223bdf04eed288b3dc',
           url=('https://download.openmmlab.com/mmpose/v1/projects/rtmpose/'
                'rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'),
           training_only=True),
)

_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--todos', action='store_true',
                   help='Inclui os pesos só de treino: RTMW-x original e RTMDet')
    p.add_argument('--origem', type=Path, default=None,
                   help='Pasta com os pesos treinados, baixada do Drive do autor')
    p.add_argument('--empacotar', type=Path, default=None,
                   help='Copia os pesos treinados desta máquina para a pasta, '
                        'com os nomes de publicação')
    return p.parse_args()


def package(destination_dir: Path) -> None:
    """Prepara os pesos treinados para publicação, conferindo cada um."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for weight in WEIGHTS:
        if weight.published_name is None:
            continue
        source = Path(weight.destination)
        if sha256_of(source) != weight.sha256:
            raise SystemExit(f'{source} não confere com o SHA-256 registrado; '
                             f'o checkpoint mudou desde a publicação')
        shutil.copy2(source, destination_dir / weight.published_name)
        print(f'  {weight.published_name}')
    print(f'pronto em {destination_dir}: suba estes arquivos na pasta do '
          f'Drive compartilhada com quem vai testar')


def fetch(weight: Weight, local_dir: Path | None) -> None:
    """Obtém um peso, se ainda não estiver no lugar, e confere o SHA-256."""
    destination = Path(weight.destination)
    if destination.exists() and sha256_of(destination) == weight.sha256:
        print(f'  já presente: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.parcial')
    if weight.url is None:
        source = (local_dir / weight.published_name) if local_dir else None
        if source is None or not source.exists():
            raise SystemExit(
                f'{weight.published_name} não encontrado. Os pesos treinados vêm '
                f'da pasta do Drive do autor: baixe-a e rode de novo com '
                f'--origem <pasta>')
        print(f'  copiando {weight.published_name}')
        shutil.copy2(source, partial)
    else:
        print(f'  baixando {weight.url}')
        urllib.request.urlretrieve(weight.url, partial)
    # Conferir antes de pôr no lugar: um download interrompido que ficasse no
    # caminho final seria carregado como modelo, e o erro apareceria longe daqui.
    if sha256_of(partial) != weight.sha256:
        partial.unlink()
        raise SystemExit(f'{destination}: SHA-256 não confere; baixe de novo')
    partial.rename(destination)


def main():
    args = parse_args()
    if args.empacotar is not None:
        package(args.empacotar)
        return
    for weight in WEIGHTS:
        if weight.training_only and not args.todos:
            continue
        fetch(weight, args.origem)
    print('pesos no lugar e conferidos')


if __name__ == '__main__':
    main()
