"""Substituto de `mmcv._ext` para quando os operadores compilados não existem.

O projeto instala o MMCV como `mmcv-lite`, sem operadores C++/CUDA: o MMCV
completo não tem wheel para o PyTorch 2.8, e compilá-lo exige o CUDA Toolkit. O
MMPose, porém, importa `mmcv.ops` ao carregar os próprios registros --- a cabeça
do ED-Pose, que este projeto não usa, puxa o módulo inteiro ---, e `mmcv.ops`
exige `mmcv._ext` já na importação. Sem este substituto nada do MMPose carrega.

Cada operador vira um objeto que só falha **se for chamado**, com a mensagem
dizendo qual. O caminho de inferência do projeto não chama nenhum; a única
exceção, o NMS do detector RTMDet, é suprida pelo torchvision em
`src/models/mmcv_ops_fallback.py`.

Este arquivo não é importado pelo projeto: `scripts/instalar_ambiente.sh` o copia
para `mmcv/_ext.py` no ambiente virtual, e só quando o `mmcv._ext` verdadeiro
não existe. Até 24/09/2026 uma cópia dele vivia só no `site-packages` da máquina
de desenvolvimento, criada à mão, e nenhuma instalação nova a reproduzia.
"""

import warnings


class _MissingOp:
    """Operador ausente: carrega sem erro, e falha com nome ao ser chamado."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"o operador compilado '{self._name}' do MMCV não está disponível "
            f"nesta instalação (mmcv-lite); ver src/models/mmcv_ext_stub.py")


def __getattr__(name: str):
    warnings.warn(f'mmcv._ext.{name} ausente: substituto que falha ao ser '
                  f'chamado', RuntimeWarning)
    return _MissingOp(name)
