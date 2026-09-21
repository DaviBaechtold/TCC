#!/usr/bin/env python
"""Mede o erro de um estimador 2D por região anatômica, com caixa de GT.

Controller. O whole-body AP é um número só para 133 keypoints e não diz qual
região quebrou. Esta medição existe para responder a uma pergunta específica: a
adaptação ao domínio veicular, que treina com peso zero em face e mãos por falta
de anotação, degradou essas regiões?

O conjunto é o COCO-WholeBody em escala de cinza, único com anotação de face e
mãos. Só entram instâncias com face e as duas mãos anotadas, e a caixa é a de
ground truth --- isola o estimador do erro do detector, como manda o protocolo
do projeto.

    python scripts/measure_region_error.py --tag etapa2 \\
        --ckpt work_dirs/rtmw_x_gray_lora/best_coco-wholebody_AP_epoch_5_merged.pth
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def main():
    args = parse_args()

    import cv2

    from src.evaluation.region_error import REGION_INDICES, normalized_errors, summarize
    from src.models.pose_pipeline import FullBodyPosePipeline

    spec = importlib.util.spec_from_file_location(
        'panel_defaults', Path(__file__).with_name('run_panel.py'))
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Flip test desligado: é a condição de operação do painel, que é onde a
    # suspeita apareceu. Ligado, mediria outra configuração.
    pipeline = FullBodyPosePipeline(
        panel._config_without_flip_test(args.cfg, WORK_DIR),
        args.ckpt, args.device, detector=None)

    instancias = usable_annotations(args.data_root / args.ann_file,
                                    args.max_images)
    print(f'{len(instancias)} instâncias com face e duas mãos anotadas')

    amostras, respostas, anotados_por_regiao = [], [], []
    for anotacao, imagem in instancias:
        frame = cv2.imread(str(args.data_root / 'val2017' / imagem['file_name']))
        if frame is None:
            continue
        x, y, largura, altura = anotacao['bbox']
        caixa = np.array([[x, y, x + largura, y + altura]], dtype=np.float32)
        resultado = pipeline(frame, boxes=caixa)
        if not resultado.num_people:
            continue

        verdade, anotados = ground_truth(anotacao)
        erros = normalized_errors(resultado.keypoints[0], verdade, anotados)
        if erros:
            amostras.append(erros)
            respostas.append({regiao: float(resultado.scores[0][indices].mean())
                              for regiao, indices in REGION_INDICES.items()})
            anotados_por_regiao.append(
                {regiao: int(anotados[indices].sum())
                 for regiao, indices in REGION_INDICES.items()})

    relatorio = {
        'tag': args.tag,
        'checkpoint': Path(args.ckpt).name,
        'conjunto': 'COCO-WholeBody val, escala de cinza, caixa de ground truth',
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
