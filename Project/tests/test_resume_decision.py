#!/usr/bin/env python
"""Verifica que `--resume` sem nada a retomar não vira retomada.

Dois modos de falha silenciosa, opostos, ambos já observados neste projeto:

1. Com `load_from` apontando para um checkpoint de outro treino, o MMEngine
   chama `resume()` sobre ele e restaura o **contador de épocas de lá**. Um
   checkpoint de trinta épocas num treino de quinze torna `while epoch <
   max_epochs` falso de saída: zero épocas, nenhum erro, um work_dir com
   aparência de treino concluído.
2. No caminho do LoRA, anular `load_from` sem ter o que retomar deixa os pesos
   aleatórios da construção, porque `init_weights` fica neutralizado.

O teste não constrói um Runner de verdade --- isso exigiria GPU e minutos. Ele
exercita a função de decisão com um objeto que expõe só o que ela lê, que é
exatamente a superfície onde o defeito morava.

Executar:  python tests/test_resume_decision.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

spec = importlib.util.spec_from_file_location(
    'train_wholebody', Path(__file__).resolve().parents[1] / 'scripts'
    / 'train_wholebody.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class FakeRunner:
    """Expõe apenas os três atributos que a decisão de retomada consulta."""

    def __init__(self, work_dir, load_from, resume):
        self.work_dir = work_dir
        self._load_from = load_from
        self._resume = resume


def main():
    with tempfile.TemporaryDirectory() as empty:
        runner = FakeRunner(empty, 'checkpoints/base.pth', resume=True)
        resuming = train._resolve_starting_point(runner)
        assert not resuming, 'declarou retomada sem ter o que retomar'
        assert runner._resume is False, (
            '`_resume` continuou ligado: o MMEngine chamaria resume() sobre o '
            'checkpoint base e restauraria o contador de épocas dele')
        assert runner._load_from == 'checkpoints/base.pth', (
            '`load_from` foi anulado; com init_weights neutralizado o treino '
            'partiria de pesos aleatórios')
        print('  sem nada a retomar: vira começo, load_from preservado  OK')

    with tempfile.TemporaryDirectory() as work_dir:
        checkpoint = Path(work_dir) / 'epoch_3.pth'
        checkpoint.touch()
        (Path(work_dir) / 'last_checkpoint').write_text(str(checkpoint))

        runner = FakeRunner(work_dir, 'checkpoints/base.pth', resume=True)
        resuming = train._resolve_starting_point(runner)
        assert resuming, 'não reconheceu o checkpoint do work_dir'
        assert runner._load_from is None, (
            '`load_from` sobreviveu: o MMEngine retomaria do checkpoint base '
            'em vez do mais recente do work_dir')
        print('  com checkpoint no work_dir: retoma dele                OK')

    runner = FakeRunner('qualquer', 'checkpoints/base.pth', resume=False)
    assert not train._resolve_starting_point(runner)
    assert runner._load_from == 'checkpoints/base.pth'
    print('  sem --resume: começa de load_from                       OK')

    print('\na decisão de retomada depende do work_dir, não da flag')


if __name__ == '__main__':
    main()
