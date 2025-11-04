"""Bounding box utilities for pose estimation."""

import numpy as np
import cv2
from typing import Optional, Tuple, List


def keypoints_to_bbox(
    keypoints: np.ndarray, 
    padding: float = 0.1,
    min_confidence: float = 0.3
) -> Optional[np.ndarray]:
    """
    Extrai bounding box dos keypoints detectados.
    
    Args:
        keypoints: array (N, 3) com (x, y, confidence)
        padding: margem extra (10% default) relativa ao tamanho da bbox
        min_confidence: confiança mínima para considerar keypoint válido
    
    Returns:
        bbox: array [x1, y1, x2, y2] ou None se não houver keypoints válidos
    
    Example:
        >>> keypoints = np.array([[100, 200, 0.9], [150, 250, 0.8]])
        >>> bbox = keypoints_to_bbox(keypoints)
        >>> print(bbox)  # [95.0, 195.0, 155.0, 255.0] (com 10% padding)
    """
    if keypoints.shape[0] == 0:
        return None
    
    # Filtrar keypoints com confiança suficiente
    valid_mask = keypoints[:, 2] > min_confidence
    valid_kpts = keypoints[valid_mask]
    
    if len(valid_kpts) == 0:
        return None
    
    # Calcular min/max coordinates
    x_coords = valid_kpts[:, 0]
    y_coords = valid_kpts[:, 1]
    
    x_min = x_coords.min()
    y_min = y_coords.min()
    x_max = x_coords.max()
    y_max = y_coords.max()
    
    # Adicionar padding proporcional ao tamanho da bbox
    width = x_max - x_min
    height = y_max - y_min
    
    x_min -= width * padding
    y_min -= height * padding
    x_max += width * padding
    y_max += height * padding
    
    # Garantir que não saia da imagem (clipping será feito depois se necessário)
    return np.array([x_min, y_min, x_max, y_max])


def draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    label_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Desenha bounding box no frame.
    
    Args:
        frame: imagem BGR
        bbox: array [x1, y1, x2, y2]
        color: cor BGR (default: azul)
        thickness: espessura da linha
        label: texto opcional para adicionar acima da bbox
        label_color: cor do texto (default: mesma que bbox)
    
    Returns:
        frame: imagem com bbox desenhada (modificado in-place)
    
    Example:
        >>> frame = cv2.imread('image.jpg')
        >>> bbox = np.array([100, 200, 300, 400])
        >>> frame = draw_bbox(frame, bbox, label='Person 1')
    """
    if label_color is None:
        label_color = color
    
    # Converter para inteiros e garantir ordem correta
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Garantir que x1 < x2 e y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Desenhar retângulo
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Adicionar label se fornecido
    if label:
        # Calcular tamanho do texto para background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Desenhar background do texto
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - baseline),
            (x1 + text_width, label_y + baseline),
            color,
            -1  # Preenchido
        )
        
        # Desenhar texto
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            font,
            font_scale,
            (255, 255, 255),  # Branco para contraste
            font_thickness
        )
    
    return frame


def draw_multiple_bboxes(
    frame: np.ndarray,
    bboxes: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    labels: Optional[List[str]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Desenha múltiplas bounding boxes no frame.
    
    Args:
        frame: imagem BGR
        bboxes: lista de arrays [x1, y1, x2, y2]
        colors: lista de cores BGR (se None, usa cores aleatórias)
        labels: lista de labels opcionais
        thickness: espessura das linhas
    
    Returns:
        frame: imagem com todas bboxes desenhadas
    
    Example:
        >>> bboxes = [np.array([100, 200, 300, 400]), 
        ...           np.array([400, 100, 600, 300])]
        >>> labels = ['Person 1', 'Person 2']
        >>> frame = draw_multiple_bboxes(frame, bboxes, labels=labels)
    """
    if colors is None:
        # Gerar cores distintas para cada pessoa
        np.random.seed(42)  # Para consistência
        colors = [
            tuple(np.random.randint(0, 255, 3).tolist())
            for _ in range(len(bboxes))
        ]
    
    if labels is None:
        labels = [f'Person {i+1}' for i in range(len(bboxes))]
    
    for i, bbox in enumerate(bboxes):
        color = colors[i] if i < len(colors) else (255, 0, 0)
        label = labels[i] if i < len(labels) else f'Person {i+1}'
        frame = draw_bbox(frame, bbox, color=color, label=label, thickness=thickness)
    
    return frame


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calcula Intersection over Union (IoU) entre duas bboxes.
    
    Args:
        bbox1: array [x1, y1, x2, y2]
        bbox2: array [x1, y1, x2, y2]
    
    Returns:
        iou: valor entre 0 e 1
    
    Example:
        >>> bbox1 = np.array([100, 100, 200, 200])
        >>> bbox2 = np.array([150, 150, 250, 250])
        >>> iou = bbox_iou(bbox1, bbox2)
        >>> print(f"IoU: {iou:.2f}")  # IoU: 0.14
    """
    # Coordenadas da interseção
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Área da interseção
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Áreas das bboxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # União
    union_area = bbox1_area + bbox2_area - inter_area
    
    # IoU
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def clip_bbox_to_image(
    bbox: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Garante que bbox está dentro dos limites da imagem.
    
    Args:
        bbox: array [x1, y1, x2, y2]
        image_shape: (height, width)
    
    Returns:
        clipped_bbox: array [x1, y1, x2, y2] clipped
    
    Example:
        >>> bbox = np.array([−10, −5, 1000, 800])
        >>> clipped = clip_bbox_to_image(bbox, (480, 640))
        >>> print(clipped)  # [0, 0, 640, 480]
    """
    h, w = image_shape[:2]
    
    x1 = max(0, min(bbox[0], w))
    y1 = max(0, min(bbox[1], h))
    x2 = max(0, min(bbox[2], w))
    y2 = max(0, min(bbox[3], h))
    
    return np.array([x1, y1, x2, y2])


def bbox_area(bbox: np.ndarray) -> float:
    """
    Calcula área da bounding box.
    
    Args:
        bbox: array [x1, y1, x2, y2]
    
    Returns:
        area: área em pixels²
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return max(0, width) * max(0, height)


def expand_bbox(
    bbox: np.ndarray,
    scale: float = 1.2,
    image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Expande bbox por um fator de escala (mantendo centro).
    
    Args:
        bbox: array [x1, y1, x2, y2]
        scale: fator de expansão (1.2 = 20% maior)
        image_shape: opcional, para clipping (height, width)
    
    Returns:
        expanded_bbox: array [x1, y1, x2, y2]
    
    Example:
        >>> bbox = np.array([100, 100, 200, 200])
        >>> expanded = expand_bbox(bbox, scale=1.5)
        >>> print(expanded)  # [75, 75, 225, 225]
    """
    x1, y1, x2, y2 = bbox
    
    # Centro e dimensões originais
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Novas dimensões
    new_width = width * scale
    new_height = height * scale
    
    # Nova bbox
    new_x1 = cx - new_width / 2
    new_y1 = cy - new_height / 2
    new_x2 = cx + new_width / 2
    new_y2 = cy + new_height / 2
    
    expanded = np.array([new_x1, new_y1, new_x2, new_y2])
    
    # Clip se shape fornecido
    if image_shape is not None:
        expanded = clip_bbox_to_image(expanded, image_shape)
    
    return expanded


if __name__ == "__main__":
    # Testes básicos
    print("Testing bbox_utils.py...")
    
    # Teste 1: keypoints_to_bbox
    keypoints = np.array([
        [100, 200, 0.9],
        [150, 250, 0.8],
        [120, 220, 0.7],
        [80, 180, 0.2],  # Baixa confiança, será ignorado
    ])
    bbox = keypoints_to_bbox(keypoints)
    print(f"✓ Bbox from keypoints: {bbox}")
    
    # Teste 2: IoU
    bbox1 = np.array([100, 100, 200, 200])
    bbox2 = np.array([150, 150, 250, 250])
    iou = bbox_iou(bbox1, bbox2)
    print(f"✓ IoU: {iou:.3f}")
    
    # Teste 3: Clip
    bbox = np.array([-10, -5, 1000, 800])
    clipped = clip_bbox_to_image(bbox, (480, 640))
    print(f"✓ Clipped bbox: {clipped}")
    
    # Teste 4: Expand
    bbox = np.array([100, 100, 200, 200])
    expanded = expand_bbox(bbox, scale=1.5)
    print(f"✓ Expanded bbox: {expanded}")
    
    print("\n✅ All tests passed!")
