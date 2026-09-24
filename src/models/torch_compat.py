"""Permite carregar checkpoints do OpenMMLab no PyTorch 2.6 ou mais novo.

Camada Model. Importar este módulo instala a correção; a operação é idempotente.

Os checkpoints do OpenMMLab guardam objetos NumPy no `meta`, e a partir do
PyTorch 2.6 o `torch.load` usa `weights_only=True` por padrão e os rejeita com
`WeightsUnpickler error: Unsupported global`. A mensagem sugere uma lista de
permissões, mas a lista precisa cobrir tudo que o `meta` carrega e varia por
checkpoint; confiar no arquivo inteiro é a escolha honesta aqui, já que ele vem
do repositório oficial ou do próprio treino.

Existia copiado em quinze arquivos antes de virar módulo.
"""

from __future__ import annotations

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    original_load = torch.load

    def load_trusting(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = load_trusting
    _installed = True


install()
