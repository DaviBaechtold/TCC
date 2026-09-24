"""Eleva uma sequência de keypoints 2D para 3D, quadro a quadro, ao vivo.

Camada Model. Recebe os 133 keypoints 2D que o Módulo 2 produz a cada frame,
mantém a janela temporal que o Módulo 3 exige e devolve a pose tridimensional
ancorada na raiz.

Três decisões aqui têm consequência direta sobre a qualidade e a latência do
sistema, e por isso são explícitas em vez de herdadas do treino:

**A entrada é levada à geometria de câmera em que o modelo foi treinado.** O
codec do MotionBERT normaliza o 2D pela largura do quadro, de modo que a escala
que a rede enxerga é `(2/largura)·(fx/Z)` — unidades normalizadas por metro. No
H3WB isso vale 0,4466; numa webcam de mesa calibrada (fx 959px, ocupante a
0,97m, quadro de 1280) vale 1,5454, isto é 3,46 vezes fora da distribuição de
treino. O DSTFormer não tem normalização de escala interna, e ignorar isso custa
caro. Medido em 20 janelas do sujeito retido S7, com o 2D de ground truth
reprojetado nessa webcam e o mesmo fator de decodificação calibrado nos dois
casos: normalizar pela largura do quadro dá 276,3mm de MPJPE na janela causal,
e mapear para a geometria de treino dá 48,0mm — praticamente os 47,5mm que o
mesmo checkpoint mede quando recebe o H3WB como ele vem.

`CameraView` corrige isso na **entrada**, projetando os pontos numa câmera
virtual com a geometria do H3WB. O fator de decodificação, que já existia, só
corrigia a **saída** — necessário, e insuficiente sozinho: com ele certo e a
entrada fora de escala, o erro ainda é quase seis vezes o do modelo.

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

import contextlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SEQUENCE_LENGTH = 16

# Escala que o decodificador aplica para levar a saída normalizada a metros.
#
# Ela **não é um número universal**: sai da geometria da câmera, e o padrão
# antigo deste módulo era a mediana do H3WB (4,478), onde a pessoa está a cinco
# metros de uma lente de 1145 pixels. Aplicá-lo ao habitáculo, onde o ocupante
# está a 0,66m de uma lente de 567 pixels, ampliava a pose em 3,8 vezes.
#
# A dedução vem do próprio codec. Ele projeta dois pontos a uma unidade de
# distância da raiz e mede a separação em pixels, que vale `2·fx/Z`; o fator é o
# inverso dela vezes dois, isto é `Z/fx`. Ver `factor_from_camera`.
H3WB_FACTOR = 4.478

# Resolução contra a qual o H3WB normaliza. Não é suposta: a matriz de
# intrínsecos do dataset vem normalizada e o `H36MWholeBodyDataset` a multiplica
# por 1000 ao derivar `f` e `c` em pixels. Ver `src/data/h3wb_dataset.py`.
H3WB_IMAGE_SIZE = 1000

# Pixels por metro na profundidade típica do H3WB. É `fx/Z`, e sai do próprio
# fator: como `fator = 1000·Z/fx`, vale `1000/fator`. Dá 223,3 px/m, coerente
# com a lente medida de 1147px e o sujeito a 5,14m.
H3WB_PIXELS_PER_METRE = H3WB_IMAGE_SIZE / H3WB_FACTOR

# Ponto principal da câmera virtual, em pixels de um quadro de 1000x1000.
#
# Média das quatro câmeras do Human3.6M, lida do `K` normalizado do próprio
# dataset (`metadata[sujeito][câmera]['K']`, valores de 0,5088 a 0,5198 em x e
# 0,5014 a 0,5155 em y) e multiplicada por 1000. Não é o centro do quadro, e a
# diferença importa: o codec normaliza em torno do centro geométrico, de modo
# que no treino o eixo óptico cai em (0,0281; 0,0134) no espaço normalizado, e
# não na origem. Mapear para o centro introduziria um deslocamento sistemático
# de meio ponto percentual que o treino nunca viu.
H3WB_PRINCIPAL_POINT_PX = (514.044, 506.700)

# Câmera do retrovisor interno do Drive&Act, dos arquivos de calibração do
# dataset — idênticos nas 29 gravações. O ocupante fica a 0,66m em mediana,
# medido na referência 3D.
DRIVEACT_FOCAL_PX = 567.0
DRIVEACT_PRINCIPAL_POINT_PX = (640.0, 512.0)
DRIVEACT_OCCUPANT_DEPTH_M = 0.664

DEFAULT_FACTOR = H3WB_FACTOR


@dataclass(frozen=True)
class CameraView:
    """Geometria da câmera que observa o ocupante.

    Imutável de propósito: é calibração, não estado. Trocar de câmera no meio de
    uma sequência mudaria a escala da entrada sem que o buffer temporal soubesse,
    e a janela misturaria duas geometrias.

    Attributes:
        focal_length_px: distância focal em pixels, da calibração.
        principal_point: (cx, cy) em pixels, da calibração.
        subject_depth_m: profundidade da raiz da pose, em metros. É a única
            grandeza que uma câmera monocular não observa --- daí a escala
            absoluta depender de conhecê-la, por calibração de cena ou por
            suposição declarada.
    """

    focal_length_px: float
    principal_point: tuple[float, float]
    subject_depth_m: float

    def __post_init__(self):
        if self.focal_length_px <= 0:
            raise ValueError('distância focal precisa ser positiva')
        if self.subject_depth_m <= 0:
            raise ValueError('profundidade precisa ser positiva')

    @property
    def pixels_per_metre(self) -> float:
        """Quantos pixels um metro ocupa na profundidade do sujeito."""
        return self.focal_length_px / self.subject_depth_m


def factor_from_camera(focal_length_px: float, root_depth_m: float) -> float:
    """Escala de decodificação a partir da geometria da câmera.

    O codec projeta dois pontos separados por duas unidades na profundidade da
    raiz e mede a separação resultante em pixels. Essa separação vale
    `2·fx/Z` com `fx` em milhares de pixels, de modo que o fator, definido como
    duas unidades divididas por ela, se reduz a `Z/fx`.

    Corrige a escala da **saída** e só ela: a entrada continua normalizada pela
    largura do quadro, numa escala que pode estar longe da do treino. Para
    corrigir os dois lados, passe uma `CameraView` ao `SequenceLifter`.

    Args:
        focal_length_px: distância focal em pixels, da calibração da câmera.
        root_depth_m: profundidade da raiz da pose, em metros.
    """
    if focal_length_px <= 0:
        raise ValueError('distância focal precisa ser positiva')
    return root_depth_m / (focal_length_px / 1000.0)


def to_training_geometry(keypoints: np.ndarray,
                         camera: CameraView) -> np.ndarray:
    """Reprojeta os keypoints numa câmera virtual com a geometria do H3WB.

    A transformação é uma homotetia em torno do ponto principal: leva o eixo
    óptico da câmera real ao da virtual e reescala o resto pela razão entre as
    duas resoluções angulares, `s = (fx/Z)_treino / (fx/Z)_vivo`. É exatamente o
    que a rede precisa, porque a projeção em perspectiva de um corpo rígido à
    profundidade Z é a mesma figura para qualquer par (fx, Z) de mesmo `fx/Z` —
    a diferença entre as duas câmeras é de escala, não de forma.

    O resultado é expresso em pixels de um quadro de 1000x1000, e não depende do
    tamanho do quadro ao vivo: é o que torna a entrada da rede invariante à
    resolução da webcam.
    """
    scale = H3WB_PIXELS_PER_METRE / camera.pixels_per_metre
    center_x, center_y = H3WB_PRINCIPAL_POINT_PX
    optical_x, optical_y = camera.principal_point

    mapped = np.empty_like(keypoints, dtype=np.float32)
    mapped[..., 0] = center_x + (keypoints[..., 0] - optical_x) * scale
    mapped[..., 1] = center_y + (keypoints[..., 1] - optical_y) * scale
    return mapped


# O codec do MotionBERT normaliza o 2D para [-1, 1] pela largura da imagem, e
# desloca o eixo vertical por h/w para preservar a razão de aspecto. Replicar a
# conta aqui, em vez de chamar o codec, é deliberado: o `encode` dele exige o
# alvo 3D, que em inferência não existe.
def _normalize(keypoints: np.ndarray, width: float, height: float) -> np.ndarray:
    normalized = keypoints.astype(np.float32) / width * 2.0
    normalized[..., 0] -= 1.0
    normalized[..., 1] -= height / width
    return normalized


# Resposta média do estimador 2D nos keypoints que ele de fato observa, medida
# sobre 150 quadros do Drive&Act. Serve de escala porque a saída do SimCC é
# magnitude de resposta, sem teto em 1, enquanto o treino do lifting vê
# confiança no intervalo [0, 1].
OBSERVED_RESPONSE = 8.30


def _normalize_confidence(scores: np.ndarray, scale: float) -> np.ndarray:
    """Leva a confiança ao intervalo [0, 1] que o treino do lifting usa.

    A escala precisa vir de quem chama porque depende da origem do número. A
    saída do SimCC é magnitude de resposta, sem teto em 1, e chega a 10; já os
    pesos de visibilidade de um dataset anotado já estão em [0, 1] e dividir de
    novo os achataria. Adivinhar pela grandeza do vetor funcionaria quase sempre,
    e falharia em silêncio no quadro em que quase nada foi detectado.
    """
    return np.clip(scores.astype(np.float32) / scale, 0.0, 1.0)


class SequenceLifter:
    """Buffer temporal mais estimador 2D→3D, com estado entre chamadas."""

    def __init__(self,
                 config: str | Path,
                 checkpoint: str | Path,
                 device: str = 'cuda:0',
                 camera: CameraView | None = None,
                 factor: float = DEFAULT_FACTOR,
                 sequence_length: int = SEQUENCE_LENGTH,
                 response_scale: float = OBSERVED_RESPONSE,
                 unobserved_confidence: float | None = None,
                 inference_dtype: 'torch.dtype | None' = None,
                 frame_stride: int = 1):
        """
        Args:
            camera: calibração da câmera. Com ela, o 2D é levado à geometria de
                treino antes de normalizar e o fator de decodificação sai dessa
                mesma geometria --- `factor` passa a ser ignorado. Sem ela, vale
                o caminho antigo: normalizar pela largura do quadro e decodificar
                com `factor`, uma escala aproximada que o painel declara como tal.
            unobserved_confidence: teto da confiança dos keypoints que o sistema
                sabe não ter observado. `None` desliga o teto.
            inference_dtype: precisão reduzida da inferência (`torch.bfloat16`
                ou `torch.float16`), por autocast. `None` mantém float32. Existe
                para a QP5 medir o que a precisão custa em acurácia e rende em
                latência no estágio que domina o caminho completo.
            frame_stride: de quantos em quantos quadros a janela é montada no
                caminho ao vivo. O lifting aprendeu contexto temporal no H3WB,
                cujas janelas têm intervalo mediano de 100ms entre quadros e
                duração mediana de 3,7s; a 30 FPS, com passo 1, a janela ao vivo
                dura 0,5s. Passo 3 dá 100ms entre quadros. A saída continua a
                cada quadro e causal: o quadro mais recente é sempre o último.
        """
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
        if frame_stride < 1:
            raise ValueError('frame_stride precisa ser ao menos 1')
        self._frame_stride = frame_stride
        # O buffer guarda todos os quadros do intervalo que a janela espaçada
        # cobre; a janela é amostrada dele de trás para a frente.
        self._window: deque[np.ndarray] = deque(
            maxlen=(sequence_length - 1) * frame_stride + 1)
        self._camera = camera
        self._factor = factor
        self._response_scale = response_scale
        self._unobserved_confidence = unobserved_confidence
        self._inference_dtype = inference_dtype
        if inference_dtype is not None:
            # As cabeças do MMPose convertem a saída para NumPy, que não tem
            # bfloat16; sem a correção a primeira predição levanta TypeError.
            from src.models import bf16_compat  # noqa: F401

        # O flip test duplica o custo e exige índices de espelhamento que só o
        # dataset conhece. Ao vivo a métrica é latência, então fica desligado --
        # e com ele desligado nem o `PoseLifter.predict` nem o
        # `MotionRegressionHead.predict` chegam a ler `flip_indices`.
        self._model.test_cfg = dict(flip_test=False)

    @property
    def warming_up(self) -> bool:
        """Verdadeiro enquanto a janela ainda não viu quadros suficientes."""
        return len(self._window) < self._window.maxlen

    def reset(self) -> None:
        """Descarta o contexto temporal. Necessário ao trocar de fonte."""
        self._window.clear()

    def _codec_geometry(self,
                        frame_size: tuple[int, int]) -> tuple[float, float, float]:
        """Quadro e fator com que o codec normaliza a entrada e decodifica a saída.

        Com câmera, a entrada já foi levada à câmera virtual do H3WB, e é contra
        ela que o codec trabalha: quadro de 1000x1000 e o fator do próprio H3WB.
        Os dois lados precisam vir da mesma geometria, senão a pose sai com a
        escala de uma e a forma da outra.
        """
        if self._camera is None:
            width, height = frame_size
            return float(width), float(height), self._factor
        return float(H3WB_IMAGE_SIZE), float(H3WB_IMAGE_SIZE), H3WB_FACTOR

    def _encode(self, keypoints: np.ndarray, scores: np.ndarray,
                frame_size: tuple[int, int],
                observed: np.ndarray | None) -> np.ndarray:
        """Monta a entrada normalizada (x, y, confiança) de um quadro."""
        width, height, _ = self._codec_geometry(frame_size)
        coordinates = (np.asarray(keypoints, dtype=np.float32)
                       if self._camera is None
                       else to_training_geometry(keypoints, self._camera))

        confidence = _normalize_confidence(scores, self._response_scale)
        # Peso zero na perda não cala o modelo, torna-o impune: o estimador 2D
        # fica confiante justamente onde não enxerga. Quando a montagem já diz
        # que a junta não está em quadro, a confiança é limitada aqui, no mesmo
        # teto que o treino do lifting viu.
        if observed is not None and self._unobserved_confidence is not None:
            confidence = np.where(
                observed, confidence,
                np.minimum(confidence, self._unobserved_confidence))

        return np.concatenate(
            [_normalize(coordinates, width, height),
             confidence.reshape(-1, 1)], axis=-1)

    def predict_windows(self, inputs: np.ndarray,
                        frame_size: tuple[int, int],
                        factors: np.ndarray) -> np.ndarray:
        """Eleva um lote de janelas já normalizadas.

        Args:
            inputs: [B, T, 133, 3] com (x, y, confiança) no espaço normalizado.
            frame_size: (largura, altura) contra a qual a entrada foi normalizada.
            factors: [B] fator de decodificação de cada janela.

        Returns:
            [B, 133, 3] em metros, relativo à raiz, do último quadro da janela.
        """
        from mmengine.structures import InstanceData

        from mmpose.structures import PoseDataSample

        width, height = frame_size
        window_length = inputs.shape[1]
        batch = torch.from_numpy(
            np.ascontiguousarray(inputs, dtype=np.float32)).to(self._device)

        # A saída bruta da cabeça está num espaço normalizado, e sozinha mede
        # metade do tamanho real: verificado em entradas do H3WB, 0,669m de
        # altura corporal contra 1,351m do ground truth. Quem restaura a escala
        # é o decodificador do codec, e ele lê `camera_param` e `factor` do
        # metainfo. Passar pelo `predict` do próprio modelo, em vez de decodificar
        # à mão, garante que a inferência ao vivo seja idêntica à da validação.
        samples = []
        for factor in np.asarray(factors, dtype=np.float32).reshape(-1):
            sample = PoseDataSample()
            sample.set_metainfo({
                'camera_param': {'w': width, 'h': height},
                'factor': np.full((window_length, 1), factor, np.float32),
            })
            sample.gt_instances = InstanceData()
            samples.append(sample)

        # A saída volta a float32 no decodificador, que converte para NumPy;
        # o autocast só muda o tipo das operações dentro da rede.
        precisao = (torch.autocast('cuda', dtype=self._inference_dtype)
                    if self._inference_dtype is not None
                    else contextlib.nullcontext())
        with torch.no_grad(), precisao:
            predicted = self._model.predict(batch, samples)

        poses = np.stack([
            np.squeeze(np.asarray(sample.pred_instances.keypoints))[-1]
            for sample in predicted
        ])
        return poses - poses[:, :1]          # ancora na raiz

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray,
                 frame_size: tuple[int, int],
                 observed: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            keypoints: [133, 2] no sistema de coordenadas do frame.
            scores: [133] resposta do estimador 2D.
            frame_size: (largura, altura) do frame.
            observed: [133] booleano do que a montagem da câmera enxerga.
                Só tem efeito com `unobserved_confidence` definido.

        Returns:
            [133, 3] em metros, relativo à raiz.
        """
        self._window.append(
            self._encode(keypoints, scores, frame_size, observed))

        # A partir do mais recente, de `frame_stride` em `frame_stride`: o
        # quadro atual é sempre o último da janela, e a leitura segue causal.
        window = list(self._window)[::-1][::self._frame_stride][::-1]
        while len(window) < self._sequence_length:
            window.insert(0, window[0])  # repete o mais antigo

        width, height, factor = self._codec_geometry(frame_size)
        return self.predict_windows(
            np.stack(window)[None], (width, height),
            np.array([factor], dtype=np.float32))[0]
