#!/usr/bin/env python
"""Latência do lifting contra profundidade e precisão (QP5).

Controller. O lifting é o estágio dominante do caminho completo --- cerca de 27
dos 47ms --- e a folga de tempo real é de 5%. A QP5 pergunta duas coisas, e as
duas se respondem sem treinar modelos novos:

1. **Profundidade contra latência.** A latência de uma rede não depende dos
   pesos, só da arquitetura; o DSTFormer é montado com 1 a 5 blocos, sem
   checkpoint, e cronometrado. Isto mede custo, não acurácia: um modelo mais
   raso precisaria ser treinado para saber quanto erra.
2. **Precisão contra latência e acurácia.** O modelo treinado, em float32,
   bfloat16 e float16, no caminho ao vivo (um quadro por chamada) e no S7 do
   H3WB.

Mede com a GPU livre --- a disputa com um treino triplica a latência.

    python scripts/benchmark_lifting.py \\
        --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v3.py \\
        --lift-ckpt work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEPTHS = (1, 2, 3, 4, 5)
ACCURACY_WINDOWS = 300
BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lift-cfg', required=True)
    p.add_argument('--lift-ckpt', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--out', type=Path, default=Path('results/lifting_latencia.json'))
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    from src.models import torch_compat  # noqa: F401
    import src.data.h3wb_dataset  # noqa: F401  registra o dataset
    from mmpose.registry import MODELS
    from src.data.h3wb_dataset import (last_frame_target,
                                       load_validation_windows, window_factor)
    from src.evaluation.throughput import describe_device, measure
    from src.models.sequence_lifter import H3WB_IMAGE_SIZE, SequenceLifter

    init_default_scope('mmpose')
    cfg = Config.fromfile(args.lift_cfg)
    janela = torch.randn(1, 16, 133, 3, device=args.device)

    # --- 1. profundidade: arquitetura sem pesos ------------------------------
    por_profundidade = {}
    for depth in DEPTHS:
        model_cfg = cfg.model.copy()
        model_cfg['backbone'] = dict(model_cfg['backbone'], depth=depth)
        model = MODELS.build(model_cfg).to(args.device).eval()
        parametros = sum(p.numel() for p in model.parameters()) / 1e6

        def forward(model=model):
            with torch.no_grad():
                return model.head.forward(model.backbone(janela))

        r = measure(forward, label=f'depth{depth}', conditions={},
                    iterations=args.iterations, warmup=args.warmup)
        por_profundidade[str(depth)] = {'mediana_ms': round(r.median_ms, 2),
                                        'p95_ms': round(r.p95_ms, 2),
                                        'parametros_M': round(parametros, 1)}
        print(f'  {depth} blocos: {r.median_ms:6.2f}ms  ({parametros:.1f}M parâmetros)')
        del model
        torch.cuda.empty_cache()

    # --- 2. precisão: modelo treinado, latência ao vivo e acurácia no S7 ------
    amostras = load_validation_windows(args.lift_cfg, ACCURACY_WINDOWS)
    alvos = np.stack([last_frame_target(s) for s in amostras])
    fatores = np.array([window_factor(s) for s in amostras], dtype=np.float32)
    entradas = np.stack([s['inputs'].numpy() for s in amostras])
    uma = entradas[:1]

    por_precisao = {}
    for nome, dtype in (('float32', None), ('bfloat16', torch.bfloat16),
                        ('float16', torch.float16)):
        lifter = SequenceLifter(args.lift_cfg, args.lift_ckpt, args.device,
                                inference_dtype=dtype)
        # O caminho ao vivo: uma janela por chamada, decodificação incluída.
        r = measure(lambda lifter=lifter: lifter.predict_windows(
                        uma, (H3WB_IMAGE_SIZE, H3WB_IMAGE_SIZE), fatores[:1]),
                    label=nome, conditions={}, iterations=args.iterations,
                    warmup=args.warmup)
        pred = np.concatenate([
            lifter.predict_windows(entradas[i:i + BATCH_SIZE],
                                   (H3WB_IMAGE_SIZE, H3WB_IMAGE_SIZE),
                                   fatores[i:i + BATCH_SIZE])
            for i in range(0, len(entradas), BATCH_SIZE)])
        mpjpe = float(np.linalg.norm(pred - alvos, axis=-1).mean() * 1000)
        por_precisao[nome] = {'mediana_ms': round(r.median_ms, 2),
                              'p95_ms': round(r.p95_ms, 2),
                              'mpjpe_mm': round(mpjpe, 2)}
        print(f'  {nome:9s}: {r.median_ms:6.2f}ms  MPJPE {mpjpe:6.2f}mm')
        del lifter
        torch.cuda.empty_cache()

    report = {
        'condicoes': {**describe_device(),
                      'config': args.lift_cfg,
                      'checkpoint': Path(args.lift_ckpt).name,
                      'janela': '16 quadros x 133 pontos, lote 1',
                      'iteracoes': args.iterations, 'aquecimento': args.warmup,
                      'acuracia': (f'{len(amostras)} janelas do S7, 2D de '
                                   'referência, quadro causal, sem espelhamento')},
        'por_profundidade': por_profundidade,
        'por_precisao': por_precisao,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'gravado em {args.out}')


if __name__ == '__main__':
    main()
