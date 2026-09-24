"""Perda do lifting temporal que respeita o peso de cada ponto.

Camada Model. Existe por dois defeitos da `MPJPEVelocityJointLoss` do MMPose
1.3.2, encontrados em sequência.

O primeiro: ela recebe da cabeça o `lifting_target_weight` e o descarta, a menos
que `use_target_weight` esteja ligado --- e o padrão é desligado. Num treino com
alvo parcial, como o do Drive&Act, em que 121 dos 133 pontos de cada janela não
têm referência e o alvo de todos cai no mesmo ponto, isso supervisiona esses
pontos contra um alvo vazio. O primeiro lifting veicular aprendeu assim a
colapsar face, mãos e pernas.

O segundo: ligado o peso, ela quebra. O termo de velocidade tem T-1 diferenças
entre quadros e é multiplicado pelo peso dos T quadros --- um erro de forma
(15 contra 16) que só aparece com sequências, que é o único uso da classe.

Esta perda mantém os três termos e os pesos da original, e corrige os dois
pontos. Com peso 1 em todo ponto ela devolve exatamente o mesmo valor que a do
MMPose, o que `tests/test_lifting_loss.py` verifica.
"""

from __future__ import annotations

import torch
from torch import nn

from mmpose.registry import MODELS

# Os mesmos da `MPJPEVelocityJointLoss`, para que trocar uma pela outra não mude
# o treino onde todo ponto tem referência.
DEFAULT_LAMBDA_SCALE = 0.5
DEFAULT_LAMBDA_VELOCITY = 20.0

# Evita divisão por zero numa janela sem nenhum ponto com referência.
_EPSILON = 1e-8


@MODELS.register_module()
class WeightedMPJPEVelocityLoss(nn.Module):
    """MPJPE + MPJPE com escala ajustada + erro de velocidade, ponderados.

    Args:
        lambda_scale: peso do termo de escala ajustada.
        lambda_velocity: peso do erro de velocidade.
        loss_weight: fator global, como nas demais perdas do MMPose.
    """

    def __init__(self,
                 lambda_scale: float = DEFAULT_LAMBDA_SCALE,
                 lambda_velocity: float = DEFAULT_LAMBDA_VELOCITY,
                 loss_weight: float = 1.0):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.lambda_velocity = lambda_velocity
        self.loss_weight = loss_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor,
                target_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            output, target: [N, T, K, 3] janela predita e referência.
            target_weight: [N, T, K, 1] peso de cada ponto; zero onde não há
                referência. Ausente, todo ponto pesa 1.
        """
        if target_weight is None:
            target_weight = torch.ones_like(output[..., :1])
        peso = target_weight.expand_as(output[..., :1])

        mpjpe = self._weighted_mean(torch.norm(output - target, dim=-1),
                                    peso[..., 0])

        # O fator de escala da original é a projeção da predição no alvo, média
        # sobre os pontos. Aqui a média é só sobre os que têm referência: um
        # ponto sem alvo não tem como dizer qual é a escala certa.
        norma_saida = (torch.sum(output ** 2, dim=-1, keepdim=True) * peso)
        norma_alvo = (torch.sum(target * output, dim=-1, keepdim=True) * peso)
        pontos = peso.sum(dim=-2, keepdim=True) + _EPSILON
        escala = ((norma_alvo.sum(dim=-2, keepdim=True) / pontos) /
                  (norma_saida.sum(dim=-2, keepdim=True) / pontos + _EPSILON))
        n_mpjpe = self._weighted_mean(
            torch.norm(escala * output - target, dim=-1), peso[..., 0])

        # Uma velocidade liga dois quadros, e só é supervisionada quando os dois
        # têm referência para aquele ponto. Usar o peso de um quadro só é o erro
        # de forma da original.
        velocidade_saida = output[:, 1:] - output[:, :-1]
        velocidade_alvo = target[:, 1:] - target[:, :-1]
        peso_velocidade = (peso[:, 1:] * peso[:, :-1])[..., 0]
        erro_velocidade = self._weighted_mean(
            torch.norm(velocidade_saida - velocidade_alvo, dim=-1),
            peso_velocidade)

        perda = (mpjpe + self.lambda_scale * n_mpjpe +
                 self.lambda_velocity * erro_velocidade)
        return self.loss_weight * perda

    @staticmethod
    def _weighted_mean(valores: torch.Tensor, pesos: torch.Tensor) -> torch.Tensor:
        """Média ponderada; com todos os pesos em 1, é a média simples."""
        return (valores * pesos).sum() / (pesos.sum() + _EPSILON)
