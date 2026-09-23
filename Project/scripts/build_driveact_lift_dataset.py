#!/usr/bin/env python
"""Emparelha o 2D do estimador real com a referência 3D do Drive&Act.

Controller. O Módulo 2 recebeu duas etapas de adaptação de domínio; o Módulo 3
não recebeu nenhuma --- ele é treinado no H3WB, com pessoas em pé num
laboratório, e aplicado a um ocupante sentado. Este script produz o insumo que
faltava para corrigir isso: pares de entrada 2D **real** e alvo tridimensional,
no domínio de aplicação.

O valor de treinar com o 2D do estimador, e não com o 2D projetado da
referência, é que a corrupção vem de graça e é a verdadeira: pernas grudadas na
borda, quadris extrapolados, confiança baixa onde a câmera não vê. As três
tentativas de simular essa corrupção estão registradas no Projeto Físico, e esta
é a primeira vez que ela não precisa ser simulada.

Duas condições que acompanham obrigatoriamente o conjunto gerado:

**A referência cobre 23 keypoints, não 133.** Face e mãos não têm alvo, e
entram com peso zero --- o que exige ensaio com o H3WB no treino, sob pena de
repetir o esquecimento medido na Etapa 3.

**Os quadros extraídos são de cinco em cinco.** Uma janela de dezesseis cobre
2,7 segundos, contra 0,3 no H3WB. A avaliação do Drive&Act sempre rodou nesse
mesmo ritmo, de modo que treino e medida são consistentes entre si, mas o painel
ao vivo roda a 30 quadros por segundo.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLIT = 'midlevel.chunks_90.split_0'
REFERENCIA = Path.home() / 'Downloads/extracted/openpose_3d'
NUM_KEYPOINTS = 133
MIN_JUNTAS_REFERENCIA = 8


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='train', choices=('train', 'val'))
    p.add_argument('--data-root', type=Path, default=Path('data/processed/driveact'))
    p.add_argument('--poses', type=Path, default=REFERENCIA)
    p.add_argument('--max-sequencias', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import cv2
    import importlib.util

    from src.data.driveact import read_pose_csv
    from src.evaluation.normalized_keypoint_error import occupant_index
    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline, PersonDetector,
                                          RTMDET_CHECKPOINT, RTMDET_CONFIG)

    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    painel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(painel)

    anotacoes = json.loads(
        (args.data_root / f'driveact_{SPLIT}.{args.split}.json').read_text())
    por_sequencia: dict[str, list[dict]] = {}
    for imagem in anotacoes['images']:
        por_sequencia.setdefault(imagem['file_id'], []).append(imagem)
    # A anotação 2D serve só para associar: dizer qual das caixas detectadas é
    # o ocupante. O alvo do treino continua sendo a referência 3D.
    anotacao_2d = {a['image_id']: np.asarray(a['keypoints'], np.float32)
                   .reshape(-1, 3) for a in anotacoes['annotations']}

    trabalho = Path('work_dirs/driveact_lift')
    trabalho.mkdir(parents=True, exist_ok=True)
    # Prende o RTMDet-nano de propósito, embora o padrão do sistema já seja o
    # YOLO26n-pose: o checkpoint veicular foi treinado sobre o 2D que este
    # detector produziu, e reconstruir o conjunto com outro mudaria a entrada de
    # treino sem que nada denunciasse. Uma diferença já existe e é deliberada:
    # o checkpoint atual foi treinado pela primeira caixa, com ~1% dos quadros
    # pareados a uma caixa espúria, e este script agora escolhe a do ocupante.
    detector = PersonDetector(RTMDET_CONFIG, RTMDET_CHECKPOINT,
                              args.device, DEFAULT_DETECTOR_SCORE)
    # O estimador da montagem de retrovisor, que é quem alimenta o lifting neste
    # domínio --- treinar com a saída de outro modelo mediria outro sistema.
    pose = FullBodyPosePipeline(
        painel._config_without_flip_test(painel.POSE_CONFIG, trabalho),
        painel.POSE_CHECKPOINT_BY_MOUNTING['retrovisor'], args.device, detector)

    sequencias = sorted(por_sequencia)
    if args.max_sequencias:
        sequencias = sequencias[:args.max_sequencias]

    guardado = {}
    sem_ocupante = 0
    for indice, file_id in enumerate(sequencias, 1):
        sujeito, run = file_id.split('/')
        csv = args.poses / sujeito / f'{run}.openpose.3d.csv'
        if not csv.exists():
            print(f'  {file_id}: sem referência 3D, ignorada')
            continue
        referencia = {q.frame_id: q for q in read_pose_csv(csv)}

        imagens = sorted(por_sequencia[file_id], key=lambda i: i['frame_id'])
        keypoints, scores, alvos, visiveis, quadros = [], [], [], [], []
        for imagem in imagens:
            if imagem['frame_id'] not in referencia:
                continue
            quadro = cv2.imread(str(args.data_root / args.split / imagem['file_name']))
            if quadro is None:
                continue
            resultado = pose(quadro)
            if not resultado.num_people or imagem['id'] not in anotacao_2d:
                continue
            ocupante = occupant_index(resultado.keypoints,
                                      anotacao_2d[imagem['id']])
            if ocupante is None:
                sem_ocupante += 1
                continue
            verdade = referencia[imagem['frame_id']]
            if int((verdade.confidence > 0).sum()) < MIN_JUNTAS_REFERENCIA:
                continue

            alvo = np.zeros((NUM_KEYPOINTS, 3), np.float32)
            visivel = np.zeros(NUM_KEYPOINTS, np.float32)
            anotados = len(verdade.points_3d)
            alvo[:anotados] = verdade.points_3d
            visivel[:anotados] = (verdade.confidence > 0).astype(np.float32)

            keypoints.append(resultado.keypoints[ocupante])
            scores.append(resultado.scores[ocupante])
            alvos.append(alvo)
            visiveis.append(visivel)
            quadros.append(imagem['frame_id'])

        if len(quadros) < 16:
            print(f'  {file_id}: {len(quadros)} quadros aproveitáveis, ignorada')
            continue
        guardado[f'{file_id}/keypoints'] = np.stack(keypoints).astype(np.float32)
        guardado[f'{file_id}/scores'] = np.stack(scores).astype(np.float32)
        guardado[f'{file_id}/target'] = np.stack(alvos)
        guardado[f'{file_id}/visible'] = np.stack(visiveis)
        guardado[f'{file_id}/frames'] = np.asarray(quadros, np.int32)
        print(f'  [{indice}/{len(sequencias)}] {file_id}: {len(quadros)} quadros')

    out = args.out or args.data_root / f'lift_{args.split}.npz'
    np.savez_compressed(out, **guardado)
    total = sum(v.shape[0] for k, v in guardado.items() if k.endswith('/frames'))
    print(f'{total} quadros de {len(guardado)//5} sequências gravados em {out}; '
          f'{sem_ocupante} descartados por nenhuma caixa corresponder ao ocupante')


if __name__ == '__main__':
    main()
