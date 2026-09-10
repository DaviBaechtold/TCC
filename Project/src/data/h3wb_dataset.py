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

    def __init__(self, seq_len: int = 1, multiple_target: int = 0, **kwargs):
        # `lazy_init` impede que as anotações sejam carregadas durante a
        # construção da classe base, o que permite restaurar `multiple_target`
        # antes de `_load_annotations` consultá-lo.
        super().__init__(seq_len=seq_len, multiple_target=0, lazy_init=True,
                         **kwargs)

        self.multiple_target = multiple_target
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
        instance_list, image_list = super()._load_annotations()
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
