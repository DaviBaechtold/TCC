"""Janelas do Drive&Act para o lifting: 2D do estimador, 3D da referência.

Camada Model. É a adaptação de domínio que o Módulo 3 nunca teve. O conjunto é
produzido por `scripts/build_driveact_lift_dataset.py` e difere do H3WB em três
pontos que o treino precisa respeitar.

**A supervisão é parcial.** A referência do Drive&Act cobre 23 keypoints
--- corpo e pés ---, e face e mãos não têm alvo. Elas entram com peso zero, e o
projeto já mediu o que peso zero faz: a Etapa 3 esqueceu face e mãos e o
whole-body AP caiu de 0,6931 para 0,2330. Por isso este dataset **só deve ser
usado em conjunto com o H3WB**, por ensaio, nunca sozinho.

**A entrada vem corrompida de verdade.** O 2D é o do estimador real sobre o
quadro real, com as pernas grudadas na borda e o quadril extrapolado. As três
tentativas de simular essa corrupção estão registradas no documento; aqui ela
não precisa ser simulada.

**A geometria é remapeada para a do treino.** O 2D medido é levado à câmera
virtual do H3WB, do mesmo modo que a inferência faz, de forma que a rede veja a
mesma escala de entrada nos dois conjuntos. Sem isso o lote misto teria duas
escalas, e a rede aprenderia a média de nenhuma.
"""

from __future__ import annotations

import numpy as np

from mmpose.registry import DATASETS

from src.models.sequence_lifter import (CameraView, DRIVEACT_FOCAL_PX,
                                        DRIVEACT_OCCUPANT_DEPTH_M,
                                        DRIVEACT_PRINCIPAL_POINT_PX,
                                        H3WB_FACTOR, H3WB_IMAGE_SIZE,
                                        H3WB_PRINCIPAL_POINT_PX,
                                        OBSERVED_RESPONSE, to_training_geometry)

NUM_KEYPOINTS = 133

# Distância focal média das quatro câmeras do Human3.6M, em pixels. É a da
# câmera virtual para a qual o 2D é remapeado, e precisa bater com o fator.
H3WB_FOCAL_PX = 1147.344


@DATASETS.register_module()
class DriveActLiftDataset:
    """Janelas deslizantes do domínio veicular, no formato do lifting.

    Args:
        ann_file: npz produzido por `build_driveact_lift_dataset.py`.
        seq_len: quadros por janela.
        window_stride: passo entre janelas.
        pipeline: transformações do MMPose, as mesmas do H3WB.
    """

    def __init__(self, ann_file: str, seq_len: int = 16,
                 window_stride: int = 8, pipeline=(), test_mode: bool = False,
                 **kwargs):
        from mmengine.dataset import Compose

        self.seq_len = seq_len
        self.window_stride = window_stride
        self.test_mode = test_mode
        self.pipeline = Compose(list(pipeline))
        self._camera = CameraView(
            focal_length_px=DRIVEACT_FOCAL_PX,
            principal_point=DRIVEACT_PRINCIPAL_POINT_PX,
            subject_depth_m=DRIVEACT_OCCUPANT_DEPTH_M)
        self._janelas = self._monta(np.load(ann_file))

    def _monta(self, dados) -> list[dict]:
        """Janelas contíguas por sequência, já na geometria de treino."""
        sequencias = sorted({chave.rsplit('/', 1)[0] for chave in dados.files})
        janelas = []
        for sequencia in sequencias:
            keypoints = dados[f'{sequencia}/keypoints']
            scores = dados[f'{sequencia}/scores']
            alvo = dados[f'{sequencia}/target']
            visivel = dados[f'{sequencia}/visible']

            # O 2D medido vai para a câmera virtual do H3WB, como na inferência.
            virtual = to_training_geometry(keypoints, self._camera)

            for inicio in range(0, len(keypoints) - self.seq_len + 1,
                                self.window_stride):
                fim = inicio + self.seq_len
                janelas.append({
                    'sequencia': sequencia,
                    'keypoints': virtual[inicio:fim],
                    'scores': scores[inicio:fim],
                    'target': alvo[inicio:fim],
                    'visible': visivel[inicio:fim],
                })
        return janelas

    def full_init(self) -> None:
        """Exigido pelo `CombinedDataset`, que inicializa cada subconjunto.

        As janelas são montadas no construtor, de modo que aqui não há o que
        fazer --- mas o método precisa existir, porque o invólucro o chama sem
        perguntar se o dataset é do arcabouço.
        """
        self._fully_initialized = True

    @property
    def metainfo(self) -> dict:
        """Layout COCO-WholeBody, o mesmo do H3WB com que ele é misturado."""
        from mmpose.datasets.datasets.utils import parse_pose_metainfo
        return parse_pose_metainfo(
            dict(from_file='configs/_base_/datasets/coco_wholebody.py'))

    def __len__(self) -> int:
        return len(self._janelas)

    def get_data_info(self, indice: int) -> dict:
        janela = self._janelas[indice]
        alvo = janela['target'].astype(np.float32)

        # A referência é absoluta na câmera do retrovisor; o alvo do codec é
        # absoluto na câmera virtual. Translada-se a pose para a profundidade
        # virtual preservando a forma, que é o que a perda mede depois do
        # ancoramento na raiz.
        raiz = alvo[:, :1].copy()
        profundidade_virtual = H3WB_FACTOR * H3WB_FOCAL_PX / 1000.0
        alvo = alvo - raiz
        alvo[..., 2] += profundidade_virtual

        visivel = janela['visible'].astype(np.float32)
        confianca = np.clip(janela['scores'] / OBSERVED_RESPONSE, 0.0, 1.0)

        return {
            'keypoints': janela['keypoints'].astype(np.float32),
            'keypoints_visible': confianca.astype(np.float32),
            'lifting_target': alvo,
            'lifting_target_visible': visivel,
            'keypoints_3d': alvo,
            'keypoints_3d_visible': visivel,
            'scale': np.zeros((1, 1), np.float32),
            'center': np.zeros((1, 2), np.float32),
            'factor': np.full((self.seq_len, 1), H3WB_FACTOR, np.float32),
            'id': indice,
            'category_id': 1,
            'iscrowd': 0,
            'camera_param': {
                'w': H3WB_IMAGE_SIZE, 'h': H3WB_IMAGE_SIZE,
                'f': np.array([H3WB_FOCAL_PX, H3WB_FOCAL_PX], np.float32),
                'c': np.array(H3WB_PRINCIPAL_POINT_PX, np.float32),
            },
            'target_img_path': [f'{janela["sequencia"]}_{indice}.jpg'],
            'img_paths': [f'{janela["sequencia"]}_{indice}.jpg'],
            'bbox': np.zeros((1, 4), np.float32),
            'bbox_score': np.ones((self.seq_len,), np.float32),
            'num_keypoints': NUM_KEYPOINTS,
            # Marca para `SimulatedEstimatorNoise` não corromper de novo.
            'corrupcao_real': True,
        }

    def __getitem__(self, indice: int):
        return self.pipeline(self.get_data_info(indice))
