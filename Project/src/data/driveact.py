"""Leitura e conversão das anotações do Drive&Act para o formato COCO-WholeBody.

O Drive&Act distribui pose apenas em 3D, no referencial da câmera, com
nomenclatura OpenPose BODY_25. Este módulo cobre a distância até o formato que o
MMPose consome: mapeia as juntas, projeta para 2D e emite anotações COCO.

Camada Model: não imprime, não desenha, não lê argumentos de linha de comando.
O controller correspondente é `scripts/convert_driveact.py`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

# Ordem dos 23 primeiros keypoints do COCO-WholeBody (17 de corpo + 6 de pés),
# mapeados para o nome da coluna correspondente no CSV do Drive&Act.
#
# O BODY_25 do OpenPose cobre exatamente esses 23: além das 17 juntas do COCO,
# ele traz lBigToe/lSmallToe/lHeel e os equivalentes à direita, que são os 6
# keypoints de pé do COCO-WholeBody. As juntas `neck` e `midHip` do BODY_25 não
# têm equivalente no COCO e ficam de fora — ainda assim são lidas, porque
# ajudam a estimar a bounding box quando os ombros não são detectados.
COCO_WHOLEBODY_TO_BODY25 = (
    'nose', 'lEye', 'rEye', 'lEar', 'rEar',
    'lShoulder', 'rShoulder', 'lElbow', 'rElbow', 'lWrist', 'rWrist',
    'lHip', 'rHip', 'lKnee', 'rKnee', 'lAnkle', 'rAnkle',
    'lBigToe', 'lSmallToe', 'lHeel', 'rBigToe', 'rSmallToe', 'rHeel',
)

NUM_MAPPED_KEYPOINTS = len(COCO_WHOLEBODY_TO_BODY25)
NUM_WHOLEBODY_KEYPOINTS = 133

# A confiança no CSV do Drive&Act vem numa escala de 0 a 100, e não de 0 a 1
# como no COCO. Verificado empiricamente: os valores observados vão de 1 a 60.
CONFIDENCE_SCALE = 100.0

# Frames com poucas juntas detectadas produzem bounding boxes degeneradas e
# anotações que atrapalham mais do que ajudam, tanto no treino quanto na
# avaliação. 8 juntas é o mínimo para definir um torso com ombros e quadris.
MIN_VISIBLE_KEYPOINTS = 8

# Margem em torno do envelope dos keypoints ao derivar a bounding box. O
# Drive&Act não anota caixa de pessoa, então ela é inferida; sem margem a caixa
# corta a silhueta exatamente nos keypoints e destoa das caixas do COCO, que
# envolvem o corpo inteiro.
BBOX_PADDING_RATIO = 0.15


@dataclass(frozen=True)
class CameraCalibration:
    """Parâmetros intrínsecos de uma câmera do Drive&Act."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    distortion: tuple[float, float, float, float, float]
    image_size: tuple[int, int]

    @classmethod
    def from_json(cls, path: Path) -> CameraCalibration:
        payload = json.loads(Path(path).read_text())
        intrinsics = payload['intrinsics']
        distortion = intrinsics['distortion']
        return cls(
            focal_length=(intrinsics['focallength']['fx'],
                          intrinsics['focallength']['fy']),
            principal_point=(intrinsics['principal_point']['cx'],
                             intrinsics['principal_point']['cy']),
            # Ordem exigida pelo OpenCV: k1, k2, p1, p2, k3.
            distortion=(distortion['k1'], distortion['k2'],
                        distortion['p1'], distortion['p2'], distortion['k3']),
            image_size=(intrinsics['img_size']['width'],
                        intrinsics['img_size']['height']),
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                        dtype=np.float64)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Projeta pontos 3D do referencial da câmera para pixels.

        Aplica o modelo de distorção radial em vez da projeção pinhole simples.
        Na câmera do retrovisor k1 = -0,2661, o que desloca uma junta a meio
        caminho da borda em cerca de 6% — erro suficiente para corromper o OKS,
        que é justamente uma métrica de precisão de localização.

        Args:
            points_3d: [N, 3] em metros, no referencial da câmera.

        Returns:
            [N, 2] em pixels.
        """
        # Os pontos já estão no referencial da câmera: os extrínsecos da câmera
        # do retrovisor são identidade, pois ela é a referência do rig.
        no_rotation = np.zeros(3, dtype=np.float64)
        no_translation = np.zeros(3, dtype=np.float64)

        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float64).reshape(-1, 1, 3),
            no_rotation, no_translation,
            self.camera_matrix, np.array(self.distortion, dtype=np.float64))
        return projected.reshape(-1, 2)


@dataclass(frozen=True)
class PoseFrame:
    """Pose de um frame, já mapeada para a ordem do COCO-WholeBody."""

    frame_id: int
    timestamp: float
    points_3d: np.ndarray   # [23, 3] em metros
    confidence: np.ndarray  # [23] em [0, 1]

    @property
    def num_visible(self) -> int:
        return int((self.confidence > 0).sum())


def read_pose_csv(path: Path) -> Iterator[PoseFrame]:
    """Lê um CSV `*.openpose.3d.csv` e devolve os frames com pose mapeada.

    Frames sem nenhuma detecção vêm com todos os campos zerados no arquivo
    original; são descartados aqui em vez de propagados como pose na origem.
    """
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            points = np.array(
                [[float(row[f'{joint}_{axis}']) for axis in 'xyz']
                 for joint in COCO_WHOLEBODY_TO_BODY25],
                dtype=np.float32)
            confidence = np.array(
                [float(row[f'{joint}_p']) for joint in COCO_WHOLEBODY_TO_BODY25],
                dtype=np.float32) / CONFIDENCE_SCALE

            if not confidence.any():
                continue

            yield PoseFrame(
                frame_id=int(row['frame_id']),
                timestamp=float(row['timestamp']),
                points_3d=points,
                confidence=confidence)


@dataclass(frozen=True)
class ActivitySegment:
    """Trecho de uma gravação rotulado com uma atividade."""

    participant_id: int
    file_id: str          # ex.: 'vp1/run2_2018-05-29-14-33-44.ids_1'
    frame_start: int
    frame_end: int
    activity: str

    def contains(self, frame_id: int) -> bool:
        return self.frame_start <= frame_id <= self.frame_end


def read_split(path: Path) -> list[ActivitySegment]:
    """Lê um CSV de split do Drive&Act.

    Os splits oficiais separam por participante, não por frame, o que evita
    vazamento de sujeito entre treino e teste. São usados como vêm, em vez de
    reparticionados, para manter os resultados comparáveis com a literatura
    que usa o mesmo dataset.
    """
    with Path(path).open(newline='') as handle:
        return [
            ActivitySegment(
                participant_id=int(row['participant_id']),
                file_id=row['file_id'],
                frame_start=int(row['frame_start']),
                frame_end=int(row['frame_end']),
                activity=row['activity'])
            for row in csv.DictReader(handle)
        ]


def activity_by_frame(segments: Sequence[ActivitySegment]) -> dict[int, str]:
    """Indexa atividade por frame, para estratificar métricas por condição."""
    labels: dict[int, str] = {}
    for segment in segments:
        for frame_id in range(segment.frame_start, segment.frame_end + 1):
            labels[frame_id] = segment.activity
    return labels


def bounding_box_from_keypoints(points_2d: np.ndarray,
                                visible: np.ndarray,
                                image_size: tuple[int, int]) -> tuple[float, ...]:
    """Deriva uma bounding box COCO (x, y, w, h) a partir dos keypoints visíveis.

    O Drive&Act não anota caixa de pessoa. Inferi-la dos keypoints é o caminho
    disponível, com a ressalva de que a caixa fica enviesada para a parte do
    corpo que foi detectada — motivo pelo qual frames com poucas juntas são
    descartados antes de chegar aqui.
    """
    visible_points = points_2d[visible]
    top_left = visible_points.min(axis=0)
    bottom_right = visible_points.max(axis=0)

    padding = (bottom_right - top_left) * BBOX_PADDING_RATIO
    top_left -= padding
    bottom_right += padding

    width, height = image_size
    x1, y1 = np.clip(top_left, [0, 0], [width, height])
    x2, y2 = np.clip(bottom_right, [0, 0], [width, height])

    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def extract_frames(video_path: Path,
                   frame_ids: Sequence[int],
                   output_dir: Path,
                   filename_prefix: str) -> dict[int, str]:
    """Extrai do vídeo apenas os frames pedidos, em escala de cinza.

    Percorre o vídeo sequencialmente em vez de usar `CAP_PROP_POS_FRAMES` por
    frame: busca aleatória em vídeo com codificação inter-frame força o decoder
    a voltar ao keyframe anterior a cada chamada, o que é ordens de grandeza
    mais lento quando se extrai milhares de frames de uma mesma gravação.

    Returns:
        Mapa de frame_id para o nome do arquivo gerado.
    """
    wanted = set(frame_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, str] = {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Não foi possível abrir o vídeo: {video_path}')

    try:
        frame_index = 0
        last_wanted = max(wanted) if wanted else -1
        while frame_index <= last_wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                filename = f'{filename_prefix}_{frame_index:06d}.jpg'
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(output_dir / filename), gray)
                written[frame_index] = filename
            frame_index += 1
    finally:
        capture.release()

    return written


# O COCO-WholeBody não guarda as 133 juntas num vetor só: ele as reparte em
# cinco campos, e o leitor do MMPose concatena `keypoints + foot_kpts +
# face_kpts + lefthand_kpts + righthand_kpts` nessa ordem. Emitir um vetor
# único de 133 produz um arquivo que parece certo e que o leitor rejeita.
WHOLEBODY_FIELD_SIZES = (
    ('keypoints', 17),       # corpo
    ('foot_kpts', 6),        # pés
    ('face_kpts', 68),
    ('lefthand_kpts', 21),
    ('righthand_kpts', 21),
)


def to_coco_keypoints(points_2d: np.ndarray,
                      confidence: np.ndarray) -> tuple[dict[str, list], int]:
    """Monta os cinco campos de keypoints do COCO-WholeBody.

    Só corpo e pés recebem valor; face e mãos ficam zeradas com visibilidade 0,
    porque o Drive&Act não as anota. Zerar em vez de omitir é o que permite
    avaliar corpo e pés no protocolo COCO-WholeBody sem que as regiões ausentes
    contem como erro.

    Returns:
        Os campos prontos para a anotação, e a contagem de juntas visíveis.
    """
    flat = [0.0] * (NUM_WHOLEBODY_KEYPOINTS * 3)
    num_visible = 0

    for index, (point, score) in enumerate(zip(points_2d, confidence)):
        if score <= 0:
            continue
        offset = index * 3
        flat[offset] = float(point[0])
        flat[offset + 1] = float(point[1])
        flat[offset + 2] = 2  # 2 = anotado e visível, convenção do COCO
        num_visible += 1

    fields = {}
    start = 0
    for name, size in WHOLEBODY_FIELD_SIZES:
        fields[name] = flat[start * 3:(start + size) * 3]
        start += size

    return fields, num_visible
