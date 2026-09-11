"""Dataset H3WB para lifting sequência-para-sequência.

Camada Model. Especializa o `H36MWholeBodyDataset` do MMPose para contornar uma
guarda da classe base que é inaplicável a ele.

`BaseMocapDataset.__init__` exige `seq_len == 1` sempre que `multiple_target`
está ativo. A restrição existe porque a construção padrão de janelas, em
`get_sequence_indices`, trata `multiple_target` como o próprio comprimento da
janela. O `H36MWholeBodyDataset`, porém, **sobrescreve** aquele método para
devolver lista vazia e monta as janelas em `_load_annotations` a partir de
`seq_len`, validando o resultado contra `multiple_target`. As duas exigências
são mutuamente contraditórias: não há valor de `seq_len` que satisfaça ambas.

Sem contorno, o H3WB fica restrito a prever um único frame por janela, o que é
incompatível com a cabeça de regressão do MotionBERT, que é sequência-para-
sequência e devolve `(B, T, K, 3)`.

A guarda é neutralizada adiando a inicialização das anotações: a classe base é
construída sem `multiple_target`, o atributo é restaurado em seguida, e só então
as anotações são carregadas.
"""

from __future__ import annotations

import os.path as osp
from pathlib import Path

import numpy as np

from mmpose.datasets.datasets.wholebody3d import H36MWholeBodyDataset
from mmpose.registry import DATASETS

# Resolução das imagens do Human3.6M. Não é arbitrária nem suposta: a matriz de
# intrínsecos do H3WB vem normalizada — o ponto principal fica em torno de 0,51
# e a distância focal em 1,145 — e o próprio `H36MWholeBodyDataset` multiplica
# esses valores por 1000 ao derivar `f` e `c` em pixels, o que fixa a base de
# normalização.
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000


def _h36m_style_path(img_path: str) -> str:
    """Reescreve o caminho no padrão de nomes que a métrica MPJPE sabe ler.

    A métrica identifica a ação de origem pelo nome do arquivo, partindo-o em
    `sujeito_acao.camera_frame.jpg`. O `H36MWholeBodyDataset` gera caminhos na
    forma `.../S1/Images/Directions.54138969/frame_000000.jpg`, cujo nome base
    não contém sujeito nem ação — o resultado é que a métrica interpreta o
    número do frame como nome da ação e reporta chaves sem sentido, do tipo
    `MPJPE_000000`.

    O agregado não é afetado, mas a decomposição por ação sim, e é ela que
    revela quais movimentos o modelo erra mais.
    """
    parts = Path(img_path).parts
    if len(parts) < 3:
        return img_path

    subject = parts[-3]                       # S1
    action_and_camera = parts[-2]             # Directions.54138969
    frame = Path(parts[-1]).stem              # frame_000000
    frame_index = frame.split('_')[-1]

    renamed = f'{subject}_{action_and_camera}_{frame_index}.jpg'
    return str(Path(img_path).parent / renamed)


@DATASETS.register_module()
class H3WBSeq2SeqDataset(H36MWholeBodyDataset):
    """H3WB com janela de `seq_len` frames e `multiple_target` alvos.

    Aceita `seq_len == multiple_target`, configuração em que a rede recebe a
    janela inteira e prevê os 3D de todos os seus frames.
    """

    def __init__(self,
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 window_stride: int | None = None,
                 **kwargs):
        # `lazy_init` impede que as anotações sejam carregadas durante a
        # construção da classe base, o que permite restaurar `multiple_target`
        # antes de `_load_annotations` consultá-lo.
        super().__init__(seq_len=seq_len, multiple_target=0, lazy_init=True,
                         **kwargs)

        self.multiple_target = multiple_target
        self.window_stride = window_stride or max(1, seq_len // 2)
        self.full_init()

    def _target_indices(self) -> list[int]:
        """Quais frames da janela são alvo, na mesma convenção do dataset base.

        Com `multiple_target` ativo a janela inteira é prevista. Sem ele, o alvo
        é o último frame no modo causal — que é o único utilizável em tempo
        real, por não depender de frames futuros — ou o central caso contrário.
        """
        if self.multiple_target:
            return list(range(self.multiple_target))
        return [-1] if self.causal else [self.seq_len // 2]

    def _build_windows(self):
        """Monta as janelas deslizantes com passo configurável e sem copiar dados.

        Substitui a construção da classe base por duas razões, ambas de memória.

        A primeira é o passo. A implementação distribuída avança de um frame por
        vez, de modo que janelas vizinhas compartilham quinze dos dezesseis
        frames e ainda assim guardam cópias independentes deles — inflação de
        dezesseis vezes para a mesma informação. O parâmetro `subset_frac`, que
        existiria para reduzir isso, é ignorado por esta classe, porque ela
        sobrescreve `get_sequence_indices` e o parâmetro só é lido lá.

        A segunda é a forma de recortar. Indexação avançada com lista de índices
        copia; fatiamento contíguo devolve uma vista sobre o array original.
        Como as janelas são intervalos contíguos, o fatiamento é equivalente e
        não aloca nada.

        Sem essas duas mudanças o conjunto de treino consumia 20 GB de RAM e o
        processo era encerrado pelo OOM killer.
        """
        instance_list = []
        instance_id = 0

        for subject in self.subjects:
            if subject not in self.ann_data:
                continue

            for action, sequence in self.ann_data[subject].items():
                frame_ids = sequence['frame_id']
                num_frames = len(frame_ids)
                if num_frames < self.seq_len:
                    continue

                for camera in self.camera_order_id:
                    if camera not in sequence:
                        continue

                    keypoints_2d = sequence[camera]['pose_2d']
                    keypoints_3d = sequence[camera]['camera_3d']
                    camera_param = self._camera_parameters(subject, camera)

                    for start in range(0, num_frames - self.seq_len + 1,
                                       self.window_stride):
                        stop = start + self.seq_len
                        window_2d = keypoints_2d[start:stop]
                        window_3d = keypoints_3d[start:stop] / 1000

                        instance_list.append({
                            'num_keypoints': window_2d.shape[1],
                            'keypoints': window_2d,
                            'keypoints_3d': window_3d,
                            'keypoints_visible': np.ones(
                                window_2d.shape[:2], dtype=np.float32),
                            'keypoints_3d_visible': np.ones(
                                window_2d.shape[:2], dtype=np.float32),
                            'scale': np.zeros((1, 1), dtype=np.float32),
                            'center': np.zeros((1, 2), dtype=np.float32),
                            'factor': np.zeros((1, 1), dtype=np.float32),
                            'id': instance_id,
                            'category_id': 1,
                            'iscrowd': 0,
                            'camera_param': camera_param,
                            'img_paths': [
                                self._frame_path(subject, action, camera,
                                                 frame_ids[i])
                                for i in range(start, stop)
                            ],
                            'lifting_target': window_3d,
                            'lifting_target_visible': np.ones(
                                window_2d.shape[:2], dtype=np.float32),
                            'bbox': self._window_bbox(subject, action, camera,
                                                      frame_ids[stop - 1]),
                            'bbox_score': np.ones((self.seq_len, ),
                                                  dtype=np.float32),
                        })
                        instance_id += 1

        return instance_list, []

    def _camera_parameters(self, subject: str, camera: str) -> dict:
        """Parâmetros da câmera no formato que as transformações 3D esperam."""
        raw = self.camera_data[subject][camera]
        intrinsics = np.asarray(raw['K'])[0]
        return {
            'K': intrinsics[:2, ...],
            'R': np.asarray(raw['R'])[0],
            'T': np.asarray(raw['T']).reshape(3, 1),
            'Distortion': np.asarray(raw['Distortion'])[0],
            'f': np.array([intrinsics[0, 0], intrinsics[1, 1]],
                          dtype=np.float32) * 1000,
            'c': np.array([intrinsics[0, 2], intrinsics[1, 2]],
                          dtype=np.float32) * 1000,
        }

    def _window_bbox(self, subject, action, camera, frame_id) -> np.ndarray:
        box = self.bboxes[(subject, action, camera, frame_id)]
        return np.array(
            [[box['x_min'], box['y_min'], box['x_max'], box['y_max']]],
            dtype=np.float32)

    def _frame_path(self, subject, action, camera, frame_id) -> str:
        return osp.join(self.data_root, 'original', subject, 'Images',
                        f'{action}.{camera}', f'frame_{frame_id}.jpg')

    def _load_annotations(self):
        """Normaliza os parâmetros de câmera para o formato que as transformações esperam.

        Duas incompatibilidades do `H36MWholeBodyDataset` distribuído precisam
        ser corrigidas aqui, e ambas só se manifestam em tempo de execução:

        A primeira é a ausência de `w` e `h`. O dataset monta `camera_param` com
        `K`, `R`, `T`, `Distortion`, `f` e `c`, mas não com a resolução da
        imagem, que o codec do MotionBERT exige para normalizar as coordenadas.

        A segunda é o tipo do contêiner. O dataset guarda os parâmetros como
        lista de um elemento, enquanto `RandomFlipAroundRoot` e o codec os
        tratam como dicionário — a verificação `'w' in camera_param` numa lista
        testa pertinência de elemento e falha, e `camera_param.update(...)` nem
        existe em listas.

        A terceira é o tipo da distância focal e do ponto principal. O dataset
        os monta como tuplas, mas `camera_to_image_coord` faz
        `camera_param['f'] / 1000.`, aritmética que exige array NumPy.

        A quarta é a ausência de `target_img_path`, que a métrica MPJPE lê para
        identificar a sequência de origem de cada predição. O dataset fornece
        apenas `img_path` e `img_paths`.
        """
        instance_list, image_list = self._build_windows()
        target_indices = self._target_indices()

        for instance in instance_list:
            instance['target_img_path'] = [
                _h36m_style_path(instance['img_paths'][index])
                for index in target_indices
            ]
            camera_param = instance['camera_param']
            if isinstance(camera_param, (list, tuple)):
                camera_param = camera_param[0]

            camera_param.setdefault('w', IMAGE_WIDTH)
            camera_param.setdefault('h', IMAGE_HEIGHT)
            for key in ('f', 'c'):
                if key in camera_param:
                    camera_param[key] = np.asarray(
                        camera_param[key], dtype=np.float32)

            instance['camera_param'] = camera_param

        return instance_list, image_list
