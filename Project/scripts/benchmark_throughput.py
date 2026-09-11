#!/usr/bin/env python
"""Mede a taxa de processamento do pipeline de pose na GPU local.

O requisito de tempo real do projeto (>= 20 FPS) é verificado aqui, e o
resultado é gravado em results/ porque sustenta afirmações do Projeto Físico.

Duas condições mudam o número em mais de um fator de dois e por isso são
explícitas na linha de comando e registradas no arquivo de saída:

    --flip-test     dobra o custo do estágio de pose. Habilitado na avaliação
                    de AP, desabilitado em operação — medir com ele ligado e
                    reportar como taxa de operação seria enganoso.
    --detector      inclui o RTMDet-nano. Numa câmera fixa no habitáculo o
                    frame inteiro pode servir de caixa única e este estágio
                    desaparece, então as duas configurações interessam.

Exemplo:
    python scripts/benchmark_throughput.py \
        --cfg configs/eval/rtmw_x_wholebody_eval.py \
        --ckpt checkpoints/rtmw-x_...pth --tag rtmwx_384x288
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DETECTOR_CONFIG = 'configs/detectors/rtmdet_nano_person_infer.py'
DETECTOR_CHECKPOINT = ('checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-'
                       '05d8511e.pth')
DETECTOR_SCORE_THRESHOLD = 0.3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--out-dir', default='results/throughput')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--flip-test', action='store_true',
                   help='Condição da avaliação de AP, não a de operação')
    p.add_argument('--detector', action='store_true',
                   help='Inclui o RTMDet-nano antes do estimador de pose')
    p.add_argument('--image', default=None,
                   help='Frame real. Obrigatório com --detector, porque ruído '
                        'não produz detecções e deixaria a pose fora da conta')
    return p.parse_args()


def main():
    args = parse_args()

    import cv2

    from src.models import torch_compat  # noqa: F401

    from mmengine.config import Config

    from src.evaluation.throughput import (describe_device, measure,
                                           synthetic_frame)
    from src.models.pose_pipeline import FullBodyPosePipeline, PersonDetector

    if args.detector and not args.image:
        raise SystemExit('--detector exige --image com um frame real')

    # O flip test vive no config, não na API de inferência; sobrescrevê-lo aqui
    # mantém uma única fonte de verdade para o resto dos parâmetros do modelo.
    cfg = Config.fromfile(args.cfg)
    cfg.model.test_cfg = dict(cfg.model.get('test_cfg', {}))
    cfg.model.test_cfg['flip_test'] = args.flip_test
    # O config remendado é artefato descartável, não evidência: vai para
    # work_dirs/ e não para results/, que guarda o que o documento cita.
    patched = Path('work_dirs/throughput') / f'{args.tag}_cfg.py'
    patched.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(patched)

    detector = (PersonDetector(DETECTOR_CONFIG, DETECTOR_CHECKPOINT,
                               args.device, DETECTOR_SCORE_THRESHOLD)
                if args.detector else None)
    pipeline = FullBodyPosePipeline(patched, args.ckpt, device=args.device,
                                    detector=detector)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f'não foi possível ler {args.image}')
    else:
        frame = synthetic_frame()

    people = pipeline(frame).num_people
    if people == 0:
        raise SystemExit('nenhuma pessoa no frame: a medição não incluiria a '
                         'pose. Use --image com um frame que contenha pessoas.')

    conditions = {
        **describe_device(),
        'config': args.cfg,
        'checkpoint': Path(args.ckpt).name,
        'input_size': list(cfg.codec['input_size']),
        'frame': f'{frame.shape[1]}x{frame.shape[0]}',
        'people_in_frame': people,
        'flip_test': args.flip_test,
        'detector': DETECTOR_CONFIG if args.detector else None,
        'batch_size': 1,
        'precision': 'fp32',
    }

    report = measure(lambda: pipeline(frame), label=args.tag,
                     conditions=conditions, iterations=args.iterations,
                     warmup=args.warmup)

    out = Path(args.out_dir) / f'{args.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    print(f'\nresultado gravado em {out}')
    requisito = 'CUMPRE' if report.fps >= 20 else 'NÃO CUMPRE'
    print(f'requisito de 20 FPS: {requisito} ({report.fps:.1f} FPS)')


if __name__ == '__main__':
    main()
