"""Detector de pessoas baseado em YOLOv11, alternativa ao RTMDet-nano.

Camada Model. Expõe a mesma chamada de `PersonDetector` --- recebe um frame,
devolve caixas --- de modo que `FullBodyPosePipeline` aceita os dois sem saber a
diferença. São duas classes com o mesmo contrato de chamada, e não uma interface
com duas implementações: não há terceiro caso à vista, e a abstração custaria
mais do que economiza.

Existe para responder à QP1, que pergunta qual configuração de detecção 2D
apresenta a melhor relação entre acurácia e latência no domínio veicular.

Nota sobre a variante escolhida: o checkpoint `-pose` também estima keypoints
corporais, e eles são **descartados** aqui. Num pipeline top-down o estimador do
segundo estágio recebe apenas o recorte definido pela caixa, de modo que
keypoints produzidos antes são trabalho computado e jogado fora. Medir a variante
`-pose` mesmo assim é deliberado: é ela que o Projeto Físico especificava, e a
comparação precisa ser com o que foi proposto.
"""

from __future__ import annotations

import numpy as np

COCO_PERSON_CLASS_ID = 0


class YoloPersonDetector:
    """Estágio 1 alternativo: localiza pessoas com YOLOv11."""

    def __init__(self, checkpoint: str, device: str, score_threshold: float):
        from ultralytics import YOLO

        self._model = YOLO(checkpoint)
        self._model.to(device)
        self._score_threshold = score_threshold

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: [H, W, 3] BGR uint8.

        Returns:
            [N, 4] em (x1, y1, x2, y2), float32.
        """
        # `verbose=False` silencia uma linha de log por frame, que a 30 FPS
        # inunda o terminal e mede mais o custo de imprimir que o de inferir.
        result = self._model.predict(frame, conf=self._score_threshold,
                                     classes=[COCO_PERSON_CLASS_ID],
                                     verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.astype(np.float32)
