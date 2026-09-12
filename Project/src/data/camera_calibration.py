"""Calibração intrínseca de câmera a partir de um tabuleiro de xadrez.

Camada Model. Produz os mesmos campos que os arquivos de calibração do
Drive&Act, de modo que o resto do sistema não precise distinguir uma câmera
calibrada por nós de uma calibrada pelos autores do dataset.

**Por que isto importa neste projeto, e não é burocracia.** A escala que converte
a saída do Módulo 3 em metros vale `Z/fx`, com `fx` vindo da calibração. Herdar
o `fx` de outro dataset ampliou a pose em 3,8 vezes no domínio veicular, e o
erro absoluto ficou em 1,4 metro sem que a métrica adotada acusasse --- ela era
invariante a escala. Sem calibração da câmera própria, a demonstração ao vivo
carrega o mesmo defeito.

O tabuleiro é o alvo clássico porque seus cantos internos são detectáveis com
precisão subpixel e sua geometria é conhecida sem medição: os quadrados são
iguais entre si, e só o lado precisa ser informado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Cantos **internos** do tabuleiro, que é o que a detecção encontra: um
# tabuleiro de 10 por 7 quadrados tem 9 por 6 cantos internos. Confundir os dois
# é o erro mais comum aqui, e faz a detecção falhar em todos os quadros.
DEFAULT_PATTERN = (9, 6)

# Abaixo disto o sistema de equações fica mal condicionado e os coeficientes de
# distorção saem instáveis, ainda que a calibração "funcione".
MIN_VIEWS = 10

# Largura em que a busca pelo tabuleiro roda; o refinamento usa a resolução
# cheia. Ver `find_corners`.
SEARCH_WIDTH = 640

# Critério de parada do refinamento subpixel dos cantos.
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CORNER_WINDOW = (11, 11)


@dataclass(frozen=True)
class Intrinsics:
    """Parâmetros intrínsecos, no formato dos arquivos do Drive&Act."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]
    width: int
    height: int
    reprojection_error: float
    views: int

    def as_driveact_json(self) -> dict:
        """Mesma estrutura dos `.calibration.json` que acompanham o Drive&Act."""
        k1, k2, p1, p2, k3 = self.distortion
        return {
            'intrinsics': {
                'focallength': {'fx': self.fx, 'fy': self.fy},
                'principal_point': {'cx': self.cx, 'cy': self.cy},
                'img_size': {'width': self.width, 'height': self.height},
                'distortion': {'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3},
            },
            # Registrado junto porque um intrínseco sem o erro de reprojeção não
            # diz se a calibração prestou.
            'quality': {
                'reprojection_error_px': self.reprojection_error,
                'views': self.views,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_driveact_json(), indent=2))


def find_corners(image: np.ndarray,
                 pattern: tuple[int, int] = DEFAULT_PATTERN
                 ) -> np.ndarray | None:
    """Localiza os cantos internos do tabuleiro, com refinamento subpixel.

    A busca roda numa cópia reduzida e o refinamento na resolução cheia. Buscar
    direto em 1280x720 custa 117ms por quadro sem tabuleiro em cena, contra 35ms
    em 640x360, e a janela de captura ficava lenta a ponto de parecer travada. O
    refinamento subpixel devolve a precisão que a redução tira, porque a busca só
    precisa achar o canto aproximado.

    Returns:
        [N, 1, 2] em pixels da imagem original, ou `None` se o tabuleiro não
        aparece inteiro.
    """
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    escala = min(1.0, SEARCH_WIDTH / gray.shape[1])
    busca = (gray if escala == 1.0 else
             cv2.resize(gray, None, fx=escala, fy=escala,
                        interpolation=cv2.INTER_AREA))
    found, corners = cv2.findChessboardCorners(
        busca, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK)
    if not found:
        return None

    corners = (corners / escala).astype(np.float32)
    return cv2.cornerSubPix(gray, corners, CORNER_WINDOW, (-1, -1),
                            CORNER_CRITERIA)


def calibrate(views: list[np.ndarray], image_size: tuple[int, int],
              pattern: tuple[int, int] = DEFAULT_PATTERN,
              square_size_m: float = 0.025) -> Intrinsics:
    """Resolve os intrínsecos a partir dos cantos detectados em várias vistas.

    Args:
        views: cantos de cada imagem, como `find_corners` devolve.
        image_size: (largura, altura) em pixels.
        square_size_m: lado do quadrado impresso, em metros. Ele fixa a escala
            do mundo; errá-lo não afeta `fx` nem a distorção, mas invalida
            qualquer distância que se meça depois.
    """
    if len(views) < MIN_VIEWS:
        raise ValueError(
            f'{len(views)} vistas é pouco para condicionar o sistema; '
            f'o mínimo é {MIN_VIEWS}')

    colunas, linhas = pattern
    modelo = np.zeros((linhas * colunas, 3), np.float32)
    modelo[:, :2] = np.mgrid[0:colunas, 0:linhas].T.reshape(-1, 2)
    modelo *= square_size_m

    erro, matriz, distorcao, _, _ = cv2.calibrateCamera(
        [modelo] * len(views), views, image_size, None, None)

    return Intrinsics(
        fx=float(matriz[0, 0]), fy=float(matriz[1, 1]),
        cx=float(matriz[0, 2]), cy=float(matriz[1, 2]),
        distortion=tuple(float(v) for v in distorcao.ravel()[:5]),
        width=image_size[0], height=image_size[1],
        reprojection_error=float(erro), views=len(views))


def chessboard_image(pattern: tuple[int, int] = DEFAULT_PATTERN,
                     square_px: int = 100, margin_px: int = 80) -> np.ndarray:
    """Gera o tabuleiro para impressão, com a contagem de cantos correta.

    Evita a busca por um PDF na internet e, mais importante, garante que o
    padrão impresso corresponda ao `pattern` que a detecção vai procurar.
    """
    colunas, linhas = pattern
    quadrados_x, quadrados_y = colunas + 1, linhas + 1
    tabuleiro = np.zeros((quadrados_y * square_px, quadrados_x * square_px),
                         np.uint8)
    for linha in range(quadrados_y):
        for coluna in range(quadrados_x):
            if (linha + coluna) % 2 == 0:
                tabuleiro[linha * square_px:(linha + 1) * square_px,
                          coluna * square_px:(coluna + 1) * square_px] = 255

    return cv2.copyMakeBorder(tabuleiro, margin_px, margin_px, margin_px,
                              margin_px, cv2.BORDER_CONSTANT, value=255)


# A4 em milímetros. O tabuleiro sai em paisagem porque dez quadrados de 25mm
# ocupam 250mm, que não cabem na largura de 210mm do retrato.
A4_LANDSCAPE_MM = (297.0, 210.0)
MILLIMETERS_PER_INCH = 25.4

# Régua impressa junto, para conferir se a impressora reescalou. É a verificação
# que separa uma calibração boa de uma que devolve números plausíveis e errados:
# "ajustar à página" encolhe o padrão em alguns por cento sem aviso, e o lado do
# quadrado informado ao script passa a estar errado na mesma proporção.
RULER_LENGTH_MM = 100.0


def chessboard_pdf(path: Path,
                   pattern: tuple[int, int] = DEFAULT_PATTERN,
                   square_size_mm: float = 25.0) -> None:
    """Grava o tabuleiro em PDF com dimensão física exata.

    O PNG serve para conferir a detecção na tela, mas não carrega escala: quem
    imprime decide o tamanho. Aqui o quadrado tem o tamanho declarado no papel,
    de modo que não é preciso medir com régua --- só conferir.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    largura_mm, altura_mm = A4_LANDSCAPE_MM
    colunas, linhas = pattern
    quadrados_x, quadrados_y = colunas + 1, linhas + 1
    tabuleiro_w = quadrados_x * square_size_mm
    tabuleiro_h = quadrados_y * square_size_mm

    if tabuleiro_w > largura_mm or tabuleiro_h > altura_mm:
        raise ValueError(
            f'tabuleiro de {tabuleiro_w:.0f}x{tabuleiro_h:.0f}mm não cabe em '
            f'A4 paisagem; reduza o lado do quadrado')

    figura = plt.figure(figsize=(largura_mm / MILLIMETERS_PER_INCH,
                                 altura_mm / MILLIMETERS_PER_INCH))
    eixo = figura.add_axes([0, 0, 1, 1])
    eixo.set_xlim(0, largura_mm)
    eixo.set_ylim(0, altura_mm)
    eixo.axis('off')

    # Centrado horizontalmente; deslocado para cima para abrir espaço ao rodapé.
    origem_x = (largura_mm - tabuleiro_w) / 2
    origem_y = altura_mm - tabuleiro_h - 8.0

    for linha in range(quadrados_y):
        for coluna in range(quadrados_x):
            if (linha + coluna) % 2:
                continue
            eixo.add_patch(Rectangle(
                (origem_x + coluna * square_size_mm,
                 origem_y + linha * square_size_mm),
                square_size_mm, square_size_mm,
                facecolor='black', edgecolor='none'))

    _draw_ruler(eixo, origem_x, origem_y - 14.0)
    eixo.text(origem_x, origem_y - 22.0,
              f'{colunas}x{linhas} cantos internos  ·  quadrado de '
              f'{square_size_mm:.0f} mm  ·  imprimir em A4 paisagem, escala 100%',
              fontsize=8, va='top')

    path.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(path, format='pdf')
    plt.close(figura)


def _draw_ruler(eixo, x: float, y: float) -> None:
    """Segmento de comprimento conhecido, para conferir a escala da impressão."""
    from matplotlib.patches import Rectangle

    eixo.add_patch(Rectangle((x, y), RULER_LENGTH_MM, 1.2,
                             facecolor='black', edgecolor='none'))
    for extremo in (x, x + RULER_LENGTH_MM):
        eixo.add_patch(Rectangle((extremo - 0.3, y - 2.0), 0.6, 5.2,
                                 facecolor='black', edgecolor='none'))
    eixo.text(x + RULER_LENGTH_MM + 4.0, y + 0.6,
              f'{RULER_LENGTH_MM:.0f} mm — confira com régua antes de calibrar',
              fontsize=8, va='center')
