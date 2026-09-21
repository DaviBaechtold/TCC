"""Pipeline top-down de estimação de pose 2D full-body.

Camada Model: encapsula detector de pessoas e estimador de keypoints, mede a
latência de cada estágio e devolve resultado estruturado. Não desenha, não
imprime, não lê argumentos de linha de comando.

Existe para eliminar a duplicação entre os vários scripts de inferência do
repositório, que reimplementavam este mesmo pipeline com pequenas variações.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mmengine.registry import DefaultScope

# Precisa vir antes de qualquer import que dispare o loader do mmengine.
from src.models import torch_compat  # noqa: F401

from mmpose.apis import inference_topdown, init_model  # noqa: E402

# Fronteiras dos blocos de keypoints no layout COCO-WholeBody.
KEYPOINT_REGIONS = {
    'body': slice(0, 17),
    'feet': slice(17, 23),
    'face': slice(23, 91),
    'left_hand': slice(91, 112),
    'right_hand': slice(112, 133),
}

COCO_PERSON_CLASS_ID = 0

# MMPose e MMDet mantêm registries separados sob escopos homônimos. O estimador
# de pose é inicializado por último e deixa o escopo global em 'mmpose', de modo
# que chamadas ao detector precisam trocar o escopo temporariamente.
_default_scope = DefaultScope.overwrite_default_scope


@dataclass
class PoseResult:
    """Saída do pipeline para um frame."""

    keypoints: np.ndarray        # [N, 133, 2] em pixels do frame original
    scores: np.ndarray           # [N, 133] confiança por keypoint
    boxes: np.ndarray            # [N, 4] no formato (x1, y1, x2, y2)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def num_people(self) -> int:
        return len(self.boxes)

    def _detected(self, min_score: float,
                  observed: np.ndarray | None) -> np.ndarray:
        """Máscara [N, 133] do que o painel conta como detectado.

        Quem chama pode passar a máscara de observabilidade, e então a contagem
        é exatamente o que o overlay desenha. Sem ela sobra o limiar sozinho,
        que conta junta encostada na borda do quadro: um print do painel exibia
        "Pes 2/6" sem que houvesse um pé desenhado em lugar nenhum.
        """
        if observed is not None:
            return observed
        return self.scores >= min_score

    def region_confidence(self, min_score: float,
                          observed: np.ndarray | None = None
                          ) -> dict[str, float]:
        """Confiança média por região anatômica, sobre o que foi detectado.

        É a única métrica de qualidade computável ao vivo: AP e MPJPE exigem
        ground truth, que não existe numa captura de webcam.
        """
        if self.num_people == 0:
            return {name: 0.0 for name in KEYPOINT_REGIONS}

        detected = self._detected(min_score, observed)
        confidence = {}
        for name, region in KEYPOINT_REGIONS.items():
            values = self.scores[:, region][detected[:, region]]
            confidence[name] = float(values.mean()) if values.size else 0.0
        return confidence

    def region_counts(self, min_score: float,
                      observed: np.ndarray | None = None
                      ) -> dict[str, tuple[int, int]]:
        """Keypoints detectados e total por região, somando todas as pessoas."""
        detected = self._detected(min_score, observed)
        counts = {}
        for name, region in KEYPOINT_REGIONS.items():
            block = detected[:, region]
            counts[name] = (int(block.sum()), int(block.size))
        return counts


# Calibrado por medição, não arbitrado: em 40 imagens do COCO val com 73 pessoas
# anotadas, 0,3 recupera todas com 84 caixas propostas, enquanto 0,5 recupera
# 60%. O excesso de caixas o estágio seguinte tolera; a omissão é irrecuperável.
DEFAULT_DETECTOR_SCORE = 0.3


class PersonDetector:
    """Estágio 1: localiza pessoas no frame.

    Opcional. Numa câmera rigidamente montada no habitáculo os ocupantes
    aparecem em regiões previsíveis, e o frame inteiro pode servir de caixa
    única — o que elimina o custo deste estágio.
    """

    def __init__(self, config: str, checkpoint: str, device: str,
                 score_threshold: float):
        from mmdet.apis import init_detector

        # O NMS do MMDet depende de operador compilado do MMCV, ausente nesta
        # instalação. O import instala o substituto baseado em torchvision.
        from src.models import mmcv_ops_fallback  # noqa: F401

        with _default_scope('mmdet'):
            self._model = init_detector(config, checkpoint, device=device)
        self._score_threshold = score_threshold

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        from mmdet.apis import inference_detector

        with _default_scope('mmdet'):
            result = inference_detector(self._model, frame)

        instances = result.pred_instances
        keep = ((instances.labels == COCO_PERSON_CLASS_ID) &
                (instances.scores >= self._score_threshold))
        boxes = instances.bboxes[keep].cpu().numpy()
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.astype(np.float32)


class FullBodyPosePipeline:
    """Detector de pessoas seguido de estimação de 133 keypoints por pessoa."""

    def __init__(self,
                 pose_config: str | Path,
                 pose_checkpoint: str | Path,
                 device: str = 'cuda:0',
                 detector: PersonDetector | None = None):
        self._pose_model = init_model(str(pose_config), str(pose_checkpoint),
                                      device=device)
        self._detector = detector

    def __call__(self, frame: np.ndarray,
                 boxes: np.ndarray | None = None) -> PoseResult:
        """
        Args:
            frame: [H, W, 3] BGR uint8.
            boxes: [N, 4] em (x1, y1, x2, y2). Quando fornecidas, o estágio de
                detecção é pulado. É o protocolo de avaliação com caixa de
                ground truth, que mede o estimador isolado do erro do detector;
                não é operável, porque em operação não existe anotação.

        Returns:
            PoseResult com keypoints já no sistema de coordenadas do frame.
        """
        started = time.perf_counter()

        if boxes is not None:
            boxes = np.asarray(boxes, dtype=np.float32)
        elif self._detector is None:
            height, width = frame.shape[:2]
            boxes = np.array([[0, 0, width, height]], dtype=np.float32)
        else:
            boxes = self._detector(frame)
        detected = time.perf_counter()

        if len(boxes) == 0:
            return PoseResult(
                keypoints=np.zeros((0, 133, 2), np.float32),
                scores=np.zeros((0, 133), np.float32),
                boxes=boxes,
                latency_ms={'detect': (detected - started) * 1e3, 'pose': 0.0})

        # inference_topdown aplica internamente o recorte, o redimensionamento e
        # a transformação afim, e devolve os keypoints já remapeados para o
        # frame original. Recortar manualmente antes desta chamada aplicaria a
        # transformação duas vezes e deslocaria os keypoints.
        samples = inference_topdown(self._pose_model, frame, bboxes=boxes)
        estimated = time.perf_counter()

        keypoints = np.stack([s.pred_instances.keypoints[0] for s in samples])
        scores = np.stack([s.pred_instances.keypoint_scores[0] for s in samples])

        return PoseResult(
            keypoints=keypoints.astype(np.float32),
            scores=scores.astype(np.float32),
            boxes=boxes,
            latency_ms={
                'detect': (detected - started) * 1e3,
                'pose': (estimated - detected) * 1e3,
            })
