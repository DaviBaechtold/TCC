"""Painel de validação do sistema de estimação de pose.

Reproduz o layout especificado no Projeto Físico: overlay 2D, visualização 3D,
métricas por região, desempenho, controles e barra de status.

Camada View: recebe resultado pronto e desenha. Toda métrica exibida chega
calculada de fora.

O painel é composto manualmente em um único array com OpenCV, em vez de usar um
toolkit gráfico. O motivo é o requisito de tempo real: o sistema precisa
sustentar 20 FPS, e a etapa de apresentação não pode competir com a inferência
pelo orçamento de 50 ms por frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

BACKGROUND = (26, 22, 20)
PANEL_FILL = (38, 33, 30)
BORDER = (70, 62, 57)
TEXT_PRIMARY = (236, 235, 232)
TEXT_MUTED = (150, 145, 140)
ACCENT = (170, 160, 60)
WARNING = (90, 160, 235)
GOOD = (120, 200, 130)

PADDING = 14
HEADER_HEIGHT = 42
STATUS_HEIGHT = 34
BUTTON_HEIGHT = 38
TITLE_HEIGHT = 24


@dataclass
class Button:
    label: str
    key: str
    active: bool = False
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    def contains(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh


@dataclass
class PanelState:
    """Tudo que o painel precisa desenhar em um frame."""

    frame: np.ndarray | None = None
    region_confidence: dict[str, float] = field(default_factory=dict)
    region_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    fps: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    num_people: int = 0
    source_label: str = ''
    # Limiar de resposta usado para contar um keypoint como detectado. Fica no
    # estado, e não numa constante, porque o painel precisa exibi-lo: a pontuação
    # do SimCC não é probabilidade e não tem teto em 1, de modo que um número
    # como "17/17 detectados" não significa nada sem o limiar ao lado.
    score_threshold: float = 0.0
    keypoints_3d: np.ndarray | None = None
    # Ângulo da vista 3D, girado continuamente para dar noção de volume: numa
    # projeção ortográfica estática a pose é ambígua em profundidade.
    azimuth: float = 0.0
    lifting_warming_up: bool = False
    calibrated: bool = False
    frame_index: int = 0
    paused: bool = False
    message: str = ''


def _put(canvas, text, origin, scale=0.46, color=TEXT_PRIMARY, thickness=1):
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(canvas, rect, title):
    """Desenha moldura e título, devolvendo a área útil interna."""
    x, y, w, h = rect
    _put(canvas, title, (x, y - 8), 0.5, TEXT_MUTED)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_FILL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    return x + PADDING, y + PADDING, w - 2 * PADDING, h - 2 * PADDING


def _fit(image, box_w, box_h):
    """Redimensiona preservando proporção, para caber na caixa."""
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


class ValidationPanel:
    """Compõe o painel completo a partir do estado de um frame."""

    REGION_LABELS = {
        'body': 'Corpo (17)',
        'feet': 'Pes (6)',
        'face': 'Face (68)',
        'left_hand': 'Mao esq. (21)',
        'right_hand': 'Mao dir. (21)',
    }

    def __init__(self, width: int = 1440, height: int = 860):
        self.width, self.height = width, height
        self.buttons = [
            Button('Play/Pause', 'space'),
            Button('Gravar', 'r'),
            Button('Salvar frame', 's'),
            Button('Esqueleto', 'k', active=True),
            Button('Sair', 'q'),
        ]
        self._layout()

    def _layout(self):
        """Calcula os retângulos uma vez, em vez de a cada frame."""
        content_top = HEADER_HEIGHT + TITLE_HEIGHT
        content_bottom = self.height - STATUS_HEIGHT - BUTTON_HEIGHT - PADDING * 2
        column_w = (self.width - PADDING * 3) // 2
        video_h = int((content_bottom - content_top - PADDING - TITLE_HEIGHT) * 0.66)
        metrics_y = content_top + video_h + PADDING + TITLE_HEIGHT
        metrics_h = content_bottom - metrics_y

        self.rect_video = (PADDING, content_top, column_w, video_h)
        self.rect_3d = (PADDING * 2 + column_w, content_top, column_w, video_h)
        self.rect_metrics_2d = (PADDING, metrics_y, column_w, metrics_h)
        self.rect_metrics_3d = (PADDING * 2 + column_w, metrics_y, column_w, metrics_h)

        button_y = content_bottom + PADDING
        button_w = (self.width - PADDING * (len(self.buttons) + 1)) // len(self.buttons)
        for index, button in enumerate(self.buttons):
            button.rect = (PADDING + index * (button_w + PADDING), button_y,
                           button_w, BUTTON_HEIGHT)

    def button_at(self, x: int, y: int) -> Button | None:
        return next((b for b in self.buttons if b.contains(x, y)), None)

    def render(self, state: PanelState) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), BACKGROUND, np.uint8)

        title = 'Sistema de Estimacao de Pose 3D Full-Body'
        _put(canvas, title, (PADDING, 27), 0.68, TEXT_PRIMARY, 1)
        # Posição medida, e não constante: um deslocamento fixo sobrepõe o
        # subtítulo ao título assim que este muda de comprimento.
        title_width = cv2.getTextSize(title, FONT, 0.68, 1)[0][0]
        _put(canvas, 'Validador em tempo real',
             (PADDING + title_width + 22, 27), 0.5, TEXT_MUTED)
        cv2.line(canvas, (0, HEADER_HEIGHT - 6), (self.width, HEADER_HEIGHT - 6),
                 BORDER, 1)

        self._draw_video(canvas, state)
        self._draw_3d(canvas, state)
        self._draw_region_metrics(canvas, state)
        self._draw_performance(canvas, state)
        self._draw_buttons(canvas)
        self._draw_status(canvas, state)
        return canvas

    def _draw_video(self, canvas, state):
        x, y, w, h = _panel(canvas, self.rect_video, 'Frame IR + Keypoints 2D')
        if state.frame is None:
            _put(canvas, 'Sem sinal da camera', (x + 10, y + 24), 0.5, TEXT_MUTED)
            return
        fitted = _fit(state.frame, w, h)
        fh, fw = fitted.shape[:2]
        offset_x, offset_y = x + (w - fw) // 2, y + (h - fh) // 2
        canvas[offset_y:offset_y + fh, offset_x:offset_x + fw] = fitted

    def _draw_3d(self, canvas, state):
        x, y, w, h = _panel(canvas, self.rect_3d, 'Visualizacao 3D')

        if state.keypoints_3d is None:
            center_y = y + h // 2
            _put(canvas, 'Sem pose 3D neste quadro',
                 (x + 10, center_y - 12), 0.5, WARNING)
            _put(canvas, 'Nenhuma pessoa detectada, ou o lifting esta desligado.',
                 (x + 10, center_y + 10), 0.44, TEXT_MUTED)
            return

        from src.visualization.skeleton3d import draw_pose_3d

        draw_pose_3d(canvas, state.keypoints_3d, (x, y), (w, h), state.azimuth)

        # A escala só é métrica quando a câmera está calibrada: sem o fator de
        # geometria o erro cresce 68%, então exibir metros seria enganoso.
        rodape = ('Escala metrica (camera calibrada)' if state.calibrated
                  else 'Forma correta, escala aproximada: camera sem calibracao')
        _put(canvas, rodape, (x + 6, y + h - 6), 0.4, TEXT_MUTED)
        if state.lifting_warming_up:
            _put(canvas, 'Buffer temporal enchendo', (x + 6, y + 18), 0.42,
                 WARNING)

    def _draw_region_metrics(self, canvas, state):
        x, y, w, h = _panel(canvas, self.rect_metrics_2d,
                            'Deteccao por regiao (confianca media)')
        if not state.region_counts:
            _put(canvas, 'Aguardando deteccao', (x + 6, y + 22), 0.46, TEXT_MUTED)
            return

        row_y = y + 18
        for name, label in self.REGION_LABELS.items():
            detected, total = state.region_counts.get(name, (0, 0))
            confidence = state.region_confidence.get(name, 0.0)
            ratio = detected / total if total else 0.0

            _put(canvas, label, (x + 6, row_y), 0.44, TEXT_MUTED)

            # A barra cede espaço às duas colunas numéricas à sua direita;
            # dimensioná-la pela largura total deixaria "conf" fora da moldura.
            bar_x = x + 130
            bar_w = w - 320
            cv2.rectangle(canvas, (bar_x, row_y - 10),
                          (bar_x + bar_w, row_y + 2), (52, 46, 42), -1)
            if ratio > 0:
                color = GOOD if ratio > 0.6 else (ACCENT if ratio > 0.25 else WARNING)
                cv2.rectangle(canvas, (bar_x, row_y - 10),
                              (bar_x + int(bar_w * ratio), row_y + 2), color, -1)

            _put(canvas, f'{detected:3d}/{total:3d}', (bar_x + bar_w + 12, row_y),
                 0.44, TEXT_PRIMARY)
            _put(canvas, f'conf {confidence:.2f}', (bar_x + bar_w + 88, row_y),
                 0.44, TEXT_MUTED)
            row_y += 26

        # A ausência de AP aqui é deliberada e precisa ficar explícita: AP exige
        # ground truth, que não existe numa captura ao vivo.
        _put(canvas, f'Resposta SimCC, nao probabilidade. Limiar {state.score_threshold:.1f}.',
             (x + 6, y + h - 6), 0.4, TEXT_MUTED)

    def _draw_performance(self, canvas, state):
        x, y, w, h = _panel(canvas, self.rect_metrics_3d,
                            'Desempenho')
        detect_ms = state.latency_ms.get('detect', 0.0)
        pose_ms = state.latency_ms.get('pose', 0.0)
        total_ms = detect_ms + pose_ms

        fps_color = GOOD if state.fps >= 20 else (ACCENT if state.fps >= 15 else WARNING)
        _put(canvas, f'{state.fps:5.1f}', (x + 6, y + 34), 1.05, fps_color, 2)
        _put(canvas, 'FPS', (x + 108, y + 34), 0.5, TEXT_MUTED)
        _put(canvas, 'meta >= 20', (x + 108, y + 14), 0.4, TEXT_MUTED)

        rows = [
            ('Latencia total', f'{total_ms:.1f} ms'),
            ('  detector', f'{detect_ms:.1f} ms'),
            ('  pose', f'{pose_ms:.1f} ms'),
            ('Pessoas detectadas', str(state.num_people)),
        ]
        row_y = y + 62
        for label, value in rows:
            _put(canvas, label, (x + 6, row_y), 0.44, TEXT_MUTED)
            _put(canvas, value, (x + w - 96, row_y), 0.44, TEXT_PRIMARY)
            row_y += 22

        _put(canvas, 'MPJPE/PA-MPJPE exigem ground truth 3D: nao ha ao vivo.',
             (x + 6, y + h - 6), 0.4, TEXT_MUTED)

    def _draw_buttons(self, canvas):
        for button in self.buttons:
            bx, by, bw, bh = button.rect
            fill = (58, 52, 46) if button.active else PANEL_FILL
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), BORDER, 1)
            text_size = cv2.getTextSize(button.label, FONT, 0.46, 1)[0]
            _put(canvas, button.label,
                 (bx + (bw - text_size[0]) // 2, by + bh // 2 + 6), 0.46,
                 TEXT_PRIMARY if button.active else TEXT_MUTED)
            _put(canvas, f'[{button.key}]', (bx + 6, by + 13), 0.34, TEXT_MUTED)

    def _draw_status(self, canvas, state):
        y = self.height - STATUS_HEIGHT
        cv2.rectangle(canvas, (0, y), (self.width, self.height), PANEL_FILL, -1)
        cv2.line(canvas, (0, y), (self.width, y), BORDER, 1)

        status = 'PAUSADO' if state.paused else 'EXECUTANDO'
        parts = [f'Fonte: {state.source_label}',
                 f'Frame: {state.frame_index}',
                 f'Estado: {status}']
        _put(canvas, '  |  '.join(parts), (PADDING, y + 22), 0.46, TEXT_MUTED)

        if state.message:
            size = cv2.getTextSize(state.message, FONT, 0.46, 1)[0]
            _put(canvas, state.message, (self.width - size[0] - PADDING, y + 22),
                 0.46, GOOD)
