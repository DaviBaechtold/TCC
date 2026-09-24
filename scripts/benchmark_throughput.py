#!/usr/bin/env python
"""Mede a taxa de processamento do pipeline de pose na GPU local.

O requisito de tempo real do projeto (>= 20 FPS) é verificado aqui, e o
resultado é gravado em results/ porque sustenta afirmações do Projeto Físico.

Duas condições mudam o número em mais de um fator de dois e por isso são
explícitas na linha de comando e registradas no arquivo de saída:

    --flip-test     dobra o custo do estágio de pose. Habilitado na avaliação
                    de AP, desabilitado em operação — medir com ele ligado e
                    reportar como taxa de operação seria enganoso.
    --detector      escolhe o detector do estágio 1. Numa câmera fixa no habitáculo o
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

from src.models.detector_config import (DEFAULT_DETECTOR,
                                        DETECTOR_CHOICES,
                                        DETECTOR_NONE)

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
    p.add_argument('--detector', default=DEFAULT_DETECTOR,
                   choices=DETECTOR_CHOICES,
                   help='Detector de pessoas do estágio 1. O padrão é o de '
                        'operação; `nenhum` mede o pipeline sem o estágio, '
                        'que numa câmera fixa no habitáculo é alternativa '
                        'real')
    p.add_argument('--image', default=None,
                   help='Frame real. Obrigatório com --detector, porque ruído '
                        'não produz detecções e deixaria a pose fora da conta')
    p.add_argument('--channels-last', action='store_true',
                   help='Layout de memória NHWC, que favorece os núcleos '
                        'tensoriais em convoluções')
    p.add_argument('--compile', action='store_true',
                   help='Compila o grafo com torch.compile. A primeira chamada '
                        'paga a compilação, por isso o aquecimento importa')
    return p.parse_args()


def main():
    args = parse_args()

    import cv2

    from src.models import torch_compat  # noqa: F401

    from mmengine.config import Config

    from src.evaluation.throughput import (describe_device, measure,
                                           synthetic_frame)
    from src.models.pose_pipeline import (DEFAULT_DETECTOR_SCORE,
                                      FullBodyPosePipeline,
                                      build_person_detector)

    if args.detector != DETECTOR_NONE and not args.image:
        raise SystemExit(f'--detector {args.detector} exige --image com um '
                         'frame real')

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

    detector = build_person_detector(args.detector, args.device,
                                     DEFAULT_DETECTOR_SCORE)
    pipeline = FullBodyPosePipeline(patched, args.ckpt, device=args.device,
                                    detector=detector)
    _apply_inference_optimizations(pipeline, args)

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
        'detector': args.detector,
        'batch_size': 1,
        'precision': 'fp32',
        'channels_last': args.channels_last,
        'compiled': args.compile,
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


def _apply_inference_optimizations(pipeline, args) -> None:
    """Aplica otimizações que não mudam a saída, só o custo de calculá-la.

    Ficam atrás de flags, e não ligadas por padrão, porque o ganho depende da
    arquitetura e do hardware: medir é o ponto deste script. Herdadas da variante
    `run_realtime_turbo.py`, removida por duplicar o pipeline inteiro para
    acrescentar estas três linhas.
    """
    import torch

    model = pipeline._pose_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.backbone = torch.compile(model.backbone, mode='max-autotune')


if __name__ == '__main__':
    main()
