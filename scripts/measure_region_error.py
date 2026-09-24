#!/usr/bin/env python
"""Mede o erro de um estimador 2D por região anatômica, com caixa de GT.

Controller. O whole-body AP é um número só para 133 keypoints e não diz qual
região quebrou. Esta medição existe para responder a uma pergunta específica: a
adaptação ao domínio veicular, que treina com peso zero em face e mãos por falta
de anotação, degradou essas regiões?

O conjunto é o COCO-WholeBody em escala de cinza, único com anotação de face e
mãos. Só entram instâncias com face e as duas mãos anotadas.

Por padrão a caixa é a de ground truth, o que isola o estimador do erro do
detector. Com `--detector` a caixa passa a vir do detector, e a medição responde
outra pergunta: **trocar o detector degrada face e mãos?** Ela existe porque a
comparação de detectores da QP1 só mede erro corporal --- o Drive&Act não anota
face nem mãos --- e adotar um detector por essa evidência repetiria o erro da
Etapa 3, que melhorou o corpo e desfez a face sem que nada medisse a face.

    python scripts/measure_region_error.py --tag etapa2 \\
        --ckpt work_dirs/rtmw_x_gray_lora/best_coco-wholebody_AP_epoch_5_merged.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import DETECTOR_CHOICES, DETECTOR_NONE

DATA_ROOT = Path('data/processed/grayscale')
ANN_FILE = 'annotations/coco_wholebody_val_v1.0.json'
WORK_DIR = Path('work_dirs/region_error')

# Blocos do COCO-WholeBody na ordem em que a anotação os guarda.
KEYPOINT_FIELDS = (('keypoints', 17), ('foot_kpts', 6), ('face_kpts', 68),
                   ('lefthand_kpts', 21), ('righthand_kpts', 21))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cfg', default='configs/eval/rtmw_x_wholebody_eval.py')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--ann-file', default=ANN_FILE)
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--detector', default=DETECTOR_NONE,
                        choices=DETECTOR_CHOICES,
                        help='Origem da caixa. O padrão dispensa o detector e '
                             'usa a de ground truth; nomear um detector mede o '
                             'efeito dele sobre cada região')
    return parser.parse_args()


def ground_truth(annotation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Monta os 133 keypoints de uma anotação e quais deles estão anotados.

    O COCO guarda um keypoint ausente como (0, 0, 0), e medir erro contra ele
    mede a distância até o canto da imagem.
    """
    pontos, anotados = [], []
    for campo, quantidade in KEYPOINT_FIELDS:
        valores = np.asarray(annotation[campo], dtype=np.float32)
        valores = valores.reshape(quantidade, 3)
        pontos.append(valores[:, :2])
        anotados.append(valores[:, 2] > 0)
    return np.concatenate(pontos), np.concatenate(anotados)


def usable_annotations(ann_file: Path, limit: int) -> list[tuple[dict, dict]]:
    """Instâncias com face e as duas mãos anotadas, uma por imagem.

    Uma por imagem porque a caixa de GT identifica a pessoa, e duas instâncias
    da mesma imagem só acrescentariam correlação à amostra.
    """
    dados = json.loads(ann_file.read_text())
    imagens = {imagem['id']: imagem for imagem in dados['images']}

    escolhidas, vistas = [], set()
    for anotacao in dados['annotations']:
        if anotacao['image_id'] in vistas:
            continue
        if not (anotacao.get('face_valid') and anotacao.get('lefthand_valid')
                and anotacao.get('righthand_valid')):
            continue
        if anotacao.get('num_keypoints', 0) < 10 or anotacao.get('iscrowd'):
            continue
        vistas.add(anotacao['image_id'])
        escolhidas.append((anotacao, imagens[anotacao['image_id']]))
        if len(escolhidas) >= limit:
            break
    return escolhidas


# Abaixo disto a caixa detectada descreve outra pessoa, ou um recorte tão
# desalinhado que a pose não é comparável com a anotação. 0,5 é o mesmo limiar
# que o protocolo do COCO usa para considerar uma detecção correta.
MIN_IOU_CORRESPONDENCIA = 0.5


def indice_da_pessoa_anotada(caixas: np.ndarray, anotada: np.ndarray,
                             iou) -> int | None:
    """Índice da caixa detectada que corresponde à pessoa anotada.

    Returns:
        O índice de maior IoU, ou `None` se nenhuma caixa alcançar
        `MIN_IOU_CORRESPONDENCIA` --- caso em que a instância é descartada em vez
        de medida contra a pessoa errada.
    """
    pontuacoes = [iou(caixa, anotada) for caixa in caixas]
    melhor = int(np.argmax(pontuacoes))
    return melhor if pontuacoes[melhor] >= MIN_IOU_CORRESPONDENCIA else None


def main():
    args = parse_args()

    import cv2

    from src.evaluation.region_error import (REGION_INDICES, normalized_errors,
                                             summarize)
    from src.models.pose_pipeline import (FullBodyPosePipeline,
                                          build_person_detector)
    from src.utils.bbox_utils import bbox_iou

    from src.models import operating_config as panel

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Flip test desligado: é a condição de operação do painel, que é onde a
    # suspeita apareceu. Ligado, mediria outra configuração.
    detector = build_person_detector(args.detector, args.device)
    pipeline = FullBodyPosePipeline(
        panel.config_without_flip_test(args.cfg, WORK_DIR),
        args.ckpt, args.device, detector=detector)

    instancias = usable_annotations(args.data_root / args.ann_file,
                                    args.max_images)
    print(f'{len(instancias)} instâncias com face e duas mãos anotadas')

    amostras, respostas, anotados_por_regiao = [], [], []
    sem_correspondencia = 0
    for anotacao, imagem in instancias:
        frame = cv2.imread(str(args.data_root / 'val2017' / imagem['file_name']))
        if frame is None:
            continue
        x, y, largura, altura = anotacao['bbox']
        caixa = np.array([x, y, x + largura, y + altura], dtype=np.float32)
        resultado = (pipeline(frame, boxes=caixa[None]) if detector is None
                     else pipeline(frame))
        if not resultado.num_people:
            continue

        # O detector devolve todas as pessoas da imagem, e a anotação descreve
        # uma. Sem casar as duas, a medição compararia a pose de uma pessoa com
        # o ground truth de outra --- e o erro resultante não teria nada a ver
        # com a qualidade do detector.
        pessoa = 0
        if detector is not None:
            pessoa = indice_da_pessoa_anotada(resultado.boxes, caixa, bbox_iou)
            if pessoa is None:
                sem_correspondencia += 1
                continue

        verdade, anotados = ground_truth(anotacao)
        erros = normalized_errors(resultado.keypoints[pessoa], verdade, anotados)
        if erros:
            amostras.append(erros)
            respostas.append(
                {regiao: float(resultado.scores[pessoa][indices].mean())
                 for regiao, indices in REGION_INDICES.items()})
            anotados_por_regiao.append(
                {regiao: int(anotados[indices].sum())
                 for regiao, indices in REGION_INDICES.items()})

    relatorio = {
        'tag': args.tag,
        'checkpoint': Path(args.ckpt).name,
        'conjunto': 'COCO-WholeBody val, escala de cinza',
        'origem_da_caixa': ('ground truth' if args.detector == DETECTOR_NONE
                            else args.detector),
        'instancias_sem_correspondencia': sem_correspondencia,
        'flip_test': False,
        'instancias': len(amostras),
        'erro_normalizado': summarize(amostras),
        'resposta_media': {regiao: round(float(np.mean(
            [r[regiao] for r in respostas])), 2) for regiao in REGION_INDICES},
        'keypoints_anotados_por_instancia': {regiao: round(float(np.mean(
            [a[regiao] for a in anotados_por_regiao])), 1)
            for regiao in REGION_INDICES},
    }

    out = args.out or Path(f'results/erro_por_regiao_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    for regiao, valores in relatorio['erro_normalizado'].items():
        print(f'  {regiao:15s} mediana {valores["mediana"]:.4f}  '
              f'média {valores["media"]:.4f}  '
              f'resposta {relatorio["resposta_media"][regiao]:.2f}')
    print(f'gravado em {out}')


if __name__ == '__main__':
    main()
