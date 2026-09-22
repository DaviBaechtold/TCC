#!/usr/bin/env python
"""Executa a bateria de validação especificada no Projeto Físico.

Controller. Os sete testes do documento tinham números espalhados por medições
avulsas; aqui eles saem de uma execução só, cada um com critério de aceite e
condição declarada, no formato que o capítulo de resultados consome.

    python scripts/run_validation_battery.py

Três testes são **citados** de medições próprias já gravadas em `results/`, com
o arquivo de origem no relatório: reexecutar a avaliação completa do COCO a cada
bateria custaria meia hora para reproduzir um número que não mudou. Os demais
são medidos na hora.

**Não rode com um treino em andamento**: a latência do Teste 2 é o produto desta
bateria, e sob disputa de GPU ela mede outra coisa.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import (DEFAULT_DETECTOR,
                                        DETECTOR_CHOICES)

VIDEO = Path('work_dirs/panel/rec_20260921_224039.mp4')
QUADROS_LATENCIA = 300
QUADROS_SOAK = 1000
DRIVEACT_FRAMES = Path('data/processed/driveact/val')
DRIVEACT_ANOTACOES = Path('data/processed/driveact/'
                          'driveact_midlevel.chunks_90.split_0.val.json')
QUADROS_OCLUSAO = 150


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', type=Path, default=VIDEO)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--detector', default=DEFAULT_DETECTOR,
                   choices=DETECTOR_CHOICES,
                   help='Detector de pessoas do estágio 1; o padrão é '
                        'o de operação')
    p.add_argument('--out', type=Path, default=Path('results/bateria_validacao.json'))
    return p.parse_args()


def carrega_painel():
    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    painel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(painel)
    return painel


def le_resultado(caminho: str, *chaves):
    """Lê um número de uma medição já gravada, ou devolve None se faltar."""
    arquivo = Path(caminho)
    if not arquivo.exists():
        return None
    dados = json.loads(arquivo.read_text())
    for chave in chaves:
        dados = dados[chave]
    return dados


def quadros_do_video(caminho: Path, limite: int, painel):
    import cv2
    captura = cv2.VideoCapture(str(caminho))
    quadros = []
    while len(quadros) < limite:
        ok, quadro = captura.read()
        if not ok:
            break
        quadros.append(painel.to_model_domain(quadro, keep_color=False))
    captura.release()
    return quadros


def main():
    args = parse_args()
    painel = carrega_painel()

    import cv2
    import torch

    from src.evaluation.validation_battery import (coordenadas_no_quadro,
                                                   fusao_temporal,
                                                   latencia_por_quadro, ocluir)
    from src.evaluation.normalized_keypoint_error import torso_length
    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                          FullBodyPosePipeline,
                                          build_person_detector)

    trabalho = Path('work_dirs/bateria')
    trabalho.mkdir(parents=True, exist_ok=True)
    detector = build_person_detector(args.detector, args.device,
                                     DEFAULT_DETECTOR_SCORE)
    pose = FullBodyPosePipeline(
        painel._config_without_flip_test(painel.POSE_CONFIG, trabalho),
        painel.POSE_CHECKPOINT, args.device, detector)

    testes: dict[str, dict] = {}

    # --- Testes 1, 1b e 4: citados de medições próprias já gravadas ---------
    ap = le_resultado('results/baselines/rtmwx_lora_gray.json', 'metrics',
                      'coco-wholebody/AP')
    ar = le_resultado('results/baselines/rtmwx_lora_gray.json', 'metrics',
                      'coco-wholebody/AR')
    testes['1_precisao_full_body'] = {
        'fonte': 'results/baselines/rtmwx_lora_gray.json',
        'condicao': 'COCO-WholeBody val em cinza, 5.000 imagens, caixa de GT, flip test ligado',
        'ap': ap, 'ar': ar, 'criterio': 'AP > 0,70 e AR > 0,75',
        'aprovado': bool(ap and ap > 0.70 and ar and ar > 0.75),
    }
    torso = le_resultado('results/baselines/rtmwx_etapa3v2_driveact.json',
                         'metrics', 'torso/normalized_mean')
    testes['1b_erro_corporal_veicular'] = {
        'fonte': 'results/baselines/rtmwx_etapa3v2_driveact.json',
        'condicao': 'Drive&Act val, 12 keypoints observáveis, erro normalizado por tronco',
        'erro_normalizado': torso, 'criterio': '< 0,040',
        'aprovado': bool(torso and torso < 0.040),
    }
    qp1 = le_resultado('results/qp1_detectores.json')
    testes['4_detector'] = {
        'fonte': 'results/qp1_detectores.json',
        'condicao': 'seis configurações sobre 1.500 quadros do Drive&Act',
        'erro_por_configuracao': {k: v['erro_normalizado'] for k, v in qp1.items()} if qp1 else None,
        'criterio': 'limiar calibrado por medição e detector melhor que quadro inteiro',
        'aprovado': bool(qp1 and qp1['RTMDet-nano']['erro_normalizado']
                         < qp1['sem detector']['erro_normalizado']),
    }

    # --- Teste 2: latência quadro a quadro ---------------------------------
    quadros = quadros_do_video(args.video, QUADROS_LATENCIA, painel)
    lifter, _ = painel.build_lifter(argumentos_padrao(painel))
    for _ in range(20):                      # aquecimento, fora da medição
        pose(quadros[0])
    testes['2_tempo_real'] = {
        'condicao': f'{len(quadros)} quadros consecutivos de {args.video.name}, '
                    'caminho completo com detector, pose e lifting, '
                    'com sincronização da GPU',
        'criterio': f'FPS >= {int(20)} e latência p90 < 100ms',
        **latencia_por_quadro(pose, quadros, lifter, torch.cuda.synchronize),
    }

    # --- Teste 3: robustez a oclusão ---------------------------------------
    # Com o checkpoint da montagem de retrovisor, e não o da mesa: medir o
    # domínio veicular com o modelo do outro domínio mede a troca de modelo, não
    # a oclusão. A primeira execução desta bateria caiu nessa.
    pose_veicular = FullBodyPosePipeline(
        painel._config_without_flip_test(painel.POSE_CONFIG, trabalho),
        painel.POSE_CHECKPOINT_BY_MOUNTING['retrovisor'], args.device, detector)
    testes['3_oclusao'] = oclusao_driveact(pose_veicular, cv2, ocluir,
                                           torso_length)
    testes['3_oclusao']['pose_checkpoint'] = Path(
        painel.POSE_CHECKPOINT_BY_MOUNTING['retrovisor']).name

    # --- Teste 5: sistema de coordenadas e confiança por região ------------
    resultado = pose(quadros[0])
    altura, largura = quadros[0].shape[:2]
    regioes = resultado.region_confidence(painel.DEFAULT_SCORE_THRESHOLD)
    testes['5_estimador'] = {
        'condicao': 'um quadro da gravação, caixa do detector',
        'criterio': 'keypoints no sistema do quadro e confiança decomposta por região',
        **coordenadas_no_quadro(resultado, (largura, altura),
                                painel.DEFAULT_SCORE_THRESHOLD),
        'regioes': {k: round(v, 2) for k, v in regioes.items()},
        'regioes_completas': sorted(regioes) == sorted(
            ['body', 'feet', 'face', 'left_hand', 'right_hand']),
    }

    # --- Teste 6: fusão temporal contra quadro único ------------------------
    keypoints, scores = [], []
    for quadro in quadros[:120]:
        r = pose(quadro)
        if r.num_people:
            keypoints.append(r.keypoints[0])
            scores.append(r.scores[0])
    testes['6_lifting'] = {
        'condicao': f'{len(keypoints)} quadros, mesma pose 2D nos dois regimes; '
                    'quadro único emulado preenchendo a janela com cópias',
        'criterio': 'a janela temporal reduz o tremor do corpo',
        **fusao_temporal(lifter, keypoints, scores, (largura, altura)),
    }

    # --- Teste 7: execução longa, memória e exceções ------------------------
    testes['7_pipeline'] = soak(pose, lifter, args.video, painel, torch)

    relatorio = {
        'pose_checkpoint': Path(painel.POSE_CHECKPOINT).name,
        'lift_checkpoint': Path(painel.LIFT_CHECKPOINT).name,
        'testes': testes,
        'aprovados': sum(1 for t in testes.values() if t.get('aprovado')),
        'total': len(testes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    for nome, teste in testes.items():
        marca = 'PASSA' if teste.get('aprovado') else 'FALHA'
        print(f'  {marca:6s} {nome}')
    print(f'{relatorio["aprovados"]} de {relatorio["total"]} aprovados; '
          f'gravado em {args.out}')


def argumentos_padrao(painel):
    """Os mesmos padrões do painel, sem passar pela linha de comando."""
    import argparse as _a
    return _a.Namespace(
        lift_cfg=painel.LIFT_CONFIG, lift_ckpt=painel.LIFT_CHECKPOINT,
        calibracao=painel.CAMERA_CALIBRATION, device='cuda:0',
        distancia=painel.DEFAULT_SUBJECT_DEPTH_M['mesa'],
        teto_confianca=painel.LIFT_UNOBSERVED_CONFIDENCE)


def oclusao_driveact(pose, cv2, ocluir, torso_length):
    """Teste 3: degradação do erro corporal sob oclusão simulada das mãos."""
    if not DRIVEACT_ANOTACOES.exists():
        return {'condicao': 'anotações do Drive&Act ausentes', 'aprovado': None}

    anotacoes = json.loads(DRIVEACT_ANOTACOES.read_text())
    por_imagem = {a['image_id']: a for a in anotacoes['annotations']}
    imagens = [i for i in anotacoes['images'] if i['id'] in por_imagem]
    passo = max(1, len(imagens) // QUADROS_OCLUSAO)

    limpos, ocluidos = [], []
    for imagem in imagens[::passo][:QUADROS_OCLUSAO]:
        quadro = cv2.imread(str(DRIVEACT_FRAMES / imagem['file_name']))
        if quadro is None:
            continue
        anotacao = por_imagem[imagem['id']]
        verdade = np.asarray(anotacao['keypoints'], np.float32).reshape(-1, 3)
        visivel = verdade[:, 2] > 0
        escala = torso_length(verdade[:, :2], visivel)
        if not escala:
            continue

        def erro(entrada):
            r = pose(entrada)
            if not r.num_people:
                return None
            d = np.linalg.norm(r.keypoints[0][:17][visivel[:17]]
                               - verdade[:17][visivel[:17], :2], axis=-1)
            return float(d.mean() / escala)

        # Oclusão sobre os punhos anotados, que é onde o volante cobre a mão.
        alvo = quadro
        for junta in (9, 10):
            if visivel[junta]:
                alvo = ocluir(alvo, verdade[junta, :2], int(escala * 0.35))

        limpo, ocluido = erro(quadro), erro(alvo)
        if limpo is not None and ocluido is not None:
            limpos.append(limpo)
            ocluidos.append(ocluido)

    if not limpos:
        return {'condicao': 'sem quadros comparáveis', 'aprovado': None}
    limpo, ocluido = float(np.mean(limpos)), float(np.mean(ocluidos))
    degradacao = (ocluido - limpo) / limpo
    return {
        'condicao': f'{len(limpos)} quadros do Drive&Act, oclusão quadrada de '
                    '0,35 comprimento de tronco sobre cada punho anotado, '
                    'da ordem de uma mão',
        'criterio': 'degradação do erro corporal < 15%',
        'erro_sem_oclusao': round(limpo, 4),
        'erro_com_oclusao': round(ocluido, 4),
        'degradacao': round(degradacao, 4),
        'aprovado': bool(degradacao < 0.15),
    }


def soak(pose, lifter, video, painel, torch):
    """Teste 7: execução longa medindo memória e exceções."""
    import cv2

    from src.evaluation.validation_battery import memoria_estavel

    captura = cv2.VideoCapture(str(video))
    amostras, excecoes, processados = [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    while processados < QUADROS_SOAK:
        ok, quadro = captura.read()
        if not ok:                            # repete o vídeo até o alvo
            captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        try:
            entrada = painel.to_model_domain(quadro, keep_color=False)
            resultado = pose(entrada)
            if resultado.num_people:
                altura, largura = entrada.shape[:2]
                lifter(resultado.keypoints[0], resultado.scores[0],
                       (largura, altura))
        except Exception:                     # noqa: BLE001 - o teste é este
            excecoes += 1
        processados += 1
        if processados % 50 == 0:
            amostras.append(torch.cuda.memory_allocated() / 2**20)
    captura.release()

    relatorio = memoria_estavel(amostras)
    relatorio.update({
        'condicao': f'{processados} quadros pelo caminho completo, memória '
                    'amostrada a cada 50',
        'criterio': 'sem exceções e crescimento de memória < 64 MiB',
        'quadros': processados,
        'excecoes': excecoes,
        'pico_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
    })
    relatorio['aprovado'] = bool(relatorio['aprovado'] and excecoes == 0)
    return relatorio


if __name__ == '__main__':
    main()
