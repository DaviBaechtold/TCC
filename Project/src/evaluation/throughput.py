"""Mede latência e taxa de processamento do pipeline de pose.

A taxa de processamento é um requisito do projeto (>= 20 FPS) e, como qualquer
afirmação quantitativa, precisa vir acompanhada da condição em que foi medida.
Este módulo existe porque um número de FPS sem condição declarada é inútil: o
mesmo modelo entrega o dobro da taxa se o `flip test` for desligado, e uma
ordem de grandeza a menos se o cronômetro for lido antes de a GPU sincronizar.

As três armadilhas que este módulo fecha:

1. **Assincronia da CUDA.** Uma chamada de inferência retorna antes de a GPU
   terminar. Sem `torch.cuda.synchronize()` mede-se o tempo de enfileirar o
   trabalho, não o de executá-lo.
2. **Aquecimento.** As primeiras iterações pagam alocação de memória, seleção
   de algoritmo da cuDNN e compilação de kernels. Incluí-las na média
   subestima a taxa de forma arbitrária.
3. **Média contra mediana.** A distribuição de latência tem cauda longa. A
   mediana descreve o regime permanente; a média é puxada por pausas do
   coletor de lixo e do escalonador.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ThroughputReport:
    """Resultado de uma medição, com as condições que a tornam reproduzível."""

    label: str
    conditions: dict
    latencies_ms: list[float] = field(repr=False)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @property
    def fps(self) -> float:
        """Taxa sustentada, derivada da mediana e não da média.

        O requisito de 20 FPS é sobre o regime permanente. Um pico isolado de
        latência não descumpre o requisito; uma mediana alta, sim.
        """
        return 1000.0 / self.median_ms

    def as_dict(self) -> dict:
        return {
            'label': self.label,
            'conditions': self.conditions,
            'median_ms': round(self.median_ms, 3),
            'p95_ms': round(self.p95_ms, 3),
            'fps': round(self.fps, 2),
            'iterations': len(self.latencies_ms),
        }


def measure(run: Callable[[], object], label: str, conditions: dict,
            iterations: int = 100, warmup: int = 20) -> ThroughputReport:
    """Cronometra `run` após aquecer, sincronizando a GPU a cada iteração."""
    import torch

    synchronize = (torch.cuda.synchronize if torch.cuda.is_available()
                   else lambda: None)

    for _ in range(warmup):
        run()
    synchronize()

    latencies = []
    for _ in range(iterations):
        started = time.perf_counter()
        run()
        synchronize()
        latencies.append((time.perf_counter() - started) * 1e3)

    return ThroughputReport(label=label,
                            conditions={**conditions, 'warmup': warmup},
                            latencies_ms=latencies)


def describe_device() -> dict:
    """Identifica o hardware, sem o que a medição não é comparável."""
    import torch

    if not torch.cuda.is_available():
        return {'device': 'cpu', 'torch': torch.__version__}
    return {
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }


def synthetic_frame(height: int = 720, width: int = 1280) -> np.ndarray:
    """Frame de ruído com as dimensões da webcam do projeto.

    Ruído em vez de imagem real porque o custo do pipeline top-down depende do
    número de caixas, não do conteúdo. Quando o detector participa da medição, a
    entrada precisa ser uma imagem real — ruído não produz detecções, e o
    estágio de pose ficaria de fora do cronômetro.
    """
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
