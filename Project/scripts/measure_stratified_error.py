#!/usr/bin/env python
"""Mede o erro corporal veicular estratificado por iluminação e oclusão.

Controller. O Projeto Físico especifica esta avaliação na Fase 2 desde o
replanejamento, e até aqui só existia o agregado --- 0,0282 de erro normalizado
por tronco, que não diz se o sistema é uniforme ou se acerta em quadro claro e
falha no escuro. Para uma aplicação que roda à noite, é a distinção que importa.

O conjunto é a partição de validação do Drive&Act (participantes vp14 e vp15,
disjuntos dos que treinaram qualquer modelo deste projeto). A caixa vem do
detector, que é a condição de operação; `--detector nenhum` usa o quadro inteiro.

    python scripts/measure_stratified_error.py \\
        --ckpt work_dirs/rtmw_x_driveact_ensaio/best_..._merged.pth

Uma ressalva que acompanha o resultado: o estrato de oclusão conta juntas que a
referência marca visíveis, e da posição de retrovisor uma junta pode faltar por
oclusão **ou** por estar fora do campo de visão. O estrato mede quanto do corpo
está disponível, não a causa da ausência.
"""

import argparse
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import (DEFAULT_DETECTOR, DETECTOR_CHOICES,
                                        DETECTOR_NONE)

FRAMES = Path('data/processed/driveact/val')
ANNOTATIONS = Path('data/processed/driveact/'
                   'driveact_midlevel.chunks_90.split_0.val.json')
WORK_DIR = Path('work_dirs/estratificado')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=None,
                   help='Estimador 2D; o padrão é o da montagem de retrovisor, '
                        'que é a montagem deste conjunto')
    p.add_argument('--cfg', default=None,
                   help='Config do estimador; o padrão é o do painel')
    p.add_argument('--frames', type=Path, default=FRAMES)
    p.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    p.add_argument('--max-images', type=int, default=1500,
                   help='Teto de quadros. O conjunto tem 20.288, e a medição '
                        'custa uma passada de detector mais pose em cada um')
    p.add_argument('--detector', default=DEFAULT_DETECTOR,
                   choices=DETECTOR_CHOICES,
                   help='Origem da caixa; o padrão é a condição de operação')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tag', default='retrovisor')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def load_panel():
    """O painel é a definição operacional do sistema; medir outra coisa não vale."""
    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    return panel


def annotated_images(annotations: Path, limit: int) -> list[tuple[dict, dict]]:
    """Pares (imagem, anotação) do Drive&Act, um por imagem.

    O Drive&Act tem um ocupante por quadro, então uma anotação por imagem é a
    totalidade do dado --- não é amostragem.
    """
    dados = json.loads(annotations.read_text())
    por_imagem = {a['image_id']: a for a in dados['annotations']
                  if not a.get('iscrowd')}

    pares = []
    for imagem in dados['images']:
        anotacao = por_imagem.get(imagem['id'])
        if anotacao is None:
            continue
        pares.append((imagem, anotacao))
        if len(pares) >= limit:
            break
    return pares


def main():
    args = parse_args()

    import cv2

    from src.evaluation.normalized_keypoint_error import instance_error
    from src.evaluation.stratified_error import (Observation, degradation,
                                                 frame_brightness, summarize)
    from src.models.pose_pipeline import (FullBodyPosePipeline,
                                          build_person_detector)

    panel = load_panel()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = args.ckpt or panel.POSE_CHECKPOINT_BY_MOUNTING['retrovisor']
    detector = build_person_detector(args.detector, args.device)
    pipeline = FullBodyPosePipeline(
        panel._config_without_flip_test(args.cfg or panel.POSE_CONFIG, WORK_DIR),
        checkpoint, args.device, detector)

    pares = annotated_images(args.annotations, args.max_images)
    print(f'{len(pares)} quadros anotados, caixa: {args.detector}')

    observacoes, sem_deteccao, sem_tronco = [], 0, 0
    for indice, (imagem, anotacao) in enumerate(pares):
        quadro = cv2.imread(str(args.frames / imagem['file_name']))
        if quadro is None:
            continue

        if detector is None:
            altura, largura = quadro.shape[:2]
            resultado = pipeline(
                quadro, boxes=np.array([[0, 0, largura, altura]],
                                       dtype=np.float32))
        else:
            resultado = pipeline(quadro)
        if resultado.num_people == 0:
            sem_deteccao += 1
            continue

        referencia = np.asarray(anotacao['keypoints'],
                                dtype=np.float32).reshape(-1, 3)
        # Uma pessoa por quadro no Drive&Act, então a primeira caixa é a dela.
        medido = instance_error(resultado.keypoints[0], referencia)
        if medido is None:
            sem_tronco += 1
            continue

        erro, visiveis = medido
        observacoes.append(Observation(error=erro,
                                       brightness=frame_brightness(quadro),
                                       visible_joints=visiveis))

        if (indice + 1) % 250 == 0:
            print(f'  {indice + 1}/{len(pares)} quadros, '
                  f'{len(observacoes)} medidos')

    relatorio = summarize(observacoes)
    relatorio.update({
        'tag': args.tag,
        'checkpoint': Path(checkpoint).name,
        'conjunto': ('Drive&Act val, partição split_0, participantes vp14 e '
                     'vp15'),
        'origem_da_caixa': ('quadro inteiro' if args.detector == DETECTOR_NONE
                            else args.detector),
        'metrica': 'erro de keypoint corporal normalizado por comprimento de tronco',
        'quadros_sem_deteccao': sem_deteccao,
        'quadros_sem_tronco': sem_tronco,
        'degradacao_escuro_vs_claro_pct': degradation(
            relatorio['iluminacao'], melhor='alto', pior='baixo'),
        'degradacao_ocluido_vs_visivel_pct': degradation(
            relatorio['oclusao'], melhor='alto', pior='baixo'),
    })

    for criterio in ('iluminacao', 'oclusao'):
        print(f'\n  {criterio}')
        for nome, dados in relatorio[criterio].items():
            print(f'    {nome:6s} faixa {str(dados["faixa"]):>16s}  '
                  f'n={dados["instancias"]:4d}  '
                  f'mediana {dados["erro_mediano"]:.4f}  '
                  f'média {dados["erro_medio"]:.4f}')
    print(f'\n  agregado: mediana {relatorio["agregado"]["erro_mediano"]:.4f}, '
          f'média {relatorio["agregado"]["erro_medio"]:.4f}')
    print(f'  degradação no escuro: '
          f'{relatorio["degradacao_escuro_vs_claro_pct"]:+.1f}%')
    print(f'  degradação sob oclusão: '
          f'{relatorio["degradacao_ocluido_vs_visivel_pct"]:+.1f}%')

    destino = args.out or Path(f'results/erro_estratificado_{args.tag}.json')
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f'\ngravado em {destino}')


if __name__ == '__main__':
    main()
