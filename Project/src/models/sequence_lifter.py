"""Eleva uma sequência de keypoints 2D para 3D, quadro a quadro, ao vivo.

Camada Model. Recebe os 133 keypoints 2D que o Módulo 2 produz a cada frame,
mantém a janela temporal que o Módulo 3 exige e devolve a pose tridimensional
ancorada na raiz.

Duas decisões aqui têm consequência direta sobre a latência do sistema, e por
isso são explícitas em vez de herdadas do treino:

**O quadro de saída é o último da janela, não o central.** A rede é
sequência-para-sequência e prevê os dezesseis quadros; escolher o central daria
mais contexto e melhor precisão, mas a 30 FPS custaria oito quadros de espera,
ou 267 ms, acima do orçamento de latência do projeto. Escolher o último torna o
estimador causal, com atraso nulo.

**A janela é preenchida por repetição enquanto enche.** A alternativa, não
devolver nada durante os primeiros quinze quadros, deixaria meio segundo de tela
vazia no início de toda captura. A saída desse intervalo é de qualidade inferior,
porque o contexto temporal é artificial, e `warming_up` permite sinalizá-lo.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

SEQUENCE_LENGTH = 16
NUM_KEYPOINTS = 133

# Escala que o decodificador aplica para levar a saída normalizada a metros. Ela
# depende da geometria da câmera e **não é recuperável de uma imagem monocular
# sem os parâmetros intrínsecos**: no H3WB, onde eles existem, varia de 2,67 a
# 5,62 com mediana 4,478. Ao vivo, sem calibração, o padrão é essa mediana, o que
# torna a pose correta em forma e apenas aproximada em escala absoluta. Para
# visualização isso basta; para medir distâncias reais, não.
DEFAULT_FACTOR = 4.478

# O codec do MotionBERT normaliza o 2D para [-1, 1] pela largura da imagem, e
# desloca o eixo vertical por h/w para preservar a razão de aspecto. Replicar a
# conta aqui, em vez de chamar o codec, é deliberado: o `encode` dele exige o
# alvo 3D, que em inferência não existe.
def _normalize(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = keypoints.astype(np.float32) / width * 2.0
    normalized[..., 0] -= 1.0
    normalized[..., 1] -= height / width
    return normalized


class SequenceLifter:
    """Buffer temporal mais estimador 2D→3D, com estado entre chamadas."""

    def __init__(self,
                 config: str | Path,
                 checkpoint: str | Path,
                 device: str = 'cuda:0',
                 sequence_length: int = SEQUENCE_LENGTH,
                 factor: float = DEFAULT_FACTOR):
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner.checkpoint import load_checkpoint

        from mmpose.registry import MODELS

        init_default_scope('mmpose')
        cfg = Config.fromfile(str(config))
        self._model = MODELS.build(cfg.model)
        load_checkpoint(self._model, str(checkpoint), map_location='cpu')
        self._model.to(device).eval()

        self._device = device
        self._sequence_length = sequence_length
        self._window: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._factor = factor
        self._flip_indices = self._model.head.decoder.__dict__.get(
            'flip_indices', list(range(NUM_KEYPOINTS)))

        # O flip test duplica o custo e exige índices de espelhamento que só o
        # dataset conhece. Ao vivo a métrica é latência, então fica desligado.
        self._model.test_cfg = dict(flip_test=False)

    @property
    def warming_up(self) -> bool:
        """Verdadeiro enquanto a janela ainda não viu quadros suficientes."""
        return len(self._window) < self._sequence_length

    def reset(self) -> None:
        """Descarta o contexto temporal. Necessário ao trocar de fonte."""
        self._window.clear()

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray,
                 frame_size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: [133, 2] no sistema de coordenadas do frame.
            scores: [133] resposta do estimador 2D.
            frame_size: (largura, altura) do frame.

        Returns:
            [133, 3] em metros, relativo à raiz.
        """
        from mmengine.structures import InstanceData

        from mmpose.structures import PoseDataSample

        width, height = frame_size
        entry = np.concatenate(
            [_normalize(keypoints, width, height),
             scores.astype(np.float32).reshape(-1, 1)], axis=-1)
        self._window.append(entry)

        window = list(self._window)
        while len(window) < self._sequence_length:
            window.insert(0, window[0])  # repete o mais antigo

        batch = torch.from_numpy(np.stack(window)).unsqueeze(0).to(self._device)

        # A saída bruta da cabeça está num espaço normalizado, e sozinha mede
        # metade do tamanho real: verificado em entradas do H3WB, 0,669m de
        # altura corporal contra 1,351m do ground truth. Quem restaura a escala
        # é o decodificador do codec, e ele lê `camera_param` e `factor` do
        # metainfo. Passar pelo `predict` do próprio modelo, em vez de decodificar
        # à mão, garante que a inferência ao vivo seja idêntica à da validação.
        sample = PoseDataSample()
        sample.set_metainfo({
            'camera_param': {'w': width, 'h': height},
            'factor': np.full((self._sequence_length, 1), self._factor, np.float32),
            'flip_indices': self._flip_indices,
        })
        sample.gt_instances = InstanceData()

        with torch.no_grad():
            predicted = self._model.predict(batch, [sample])

        target_frame = np.squeeze(
            np.asarray(predicted[0].pred_instances.keypoints))[-1]
        return target_frame - target_frame[:1]  # ancora na raiz
