"""Adaptação de baixo posto (LoRA) para convoluções e camadas lineares.

Camada Model. Serve à adaptação do domínio RGB para escala de cinza sem
retreinar os pesos originais.

Motivação medida, e não teórica: o fine-tuning completo do RTMPose-m a uma taxa
de aprendizado de 5e-4 fez a métrica cair de 0,5255 para 0,5137 em dez épocas,
ou seja, o treino piorou o modelo em relação a não treinar. A causa é
catastrophic forgetting — a atualização destrói representações aprendidas em
centenas de horas de treino prévio para acomodar um deslocamento de domínio
comparativamente pequeno.

O LoRA contorna isso congelando os pesos originais e aprendendo apenas um termo
aditivo de posto baixo, `B @ A`, com `B` inicializado em zero. Duas
consequências: no início do treino o modelo é exatamente o original, e o
conhecimento prévio permanece recuperável, porque nunca foi sobrescrito.

Implementação própria em vez da biblioteca `peft`: são poucas dezenas de linhas,
`peft` pressupõe modelos no formato Hugging Face e traz maquinaria de
serialização e quantização que não se aplica aqui, e manter o adaptador sob
controle direto permite verificar a identidade com o modelo base.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class LoRAConv2d(nn.Module):
    """Envolve uma Conv2d congelada com um termo aditivo de posto baixo.

    A decomposição usa uma convolução `A` com o mesmo kernel da original,
    projetando para `rank` canais, seguida de uma convolução ponto-a-ponto `B`
    que retorna ao número de canais de saída. O custo de parâmetros cai de
    `Cout*Cin*k*k` para `rank*(Cin*k*k + Cout)`.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.down = nn.Conv2d(
            base.in_channels, rank, base.kernel_size,
            stride=base.stride, padding=base.padding,
            dilation=base.dilation, groups=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.scaling = alpha / rank

        # `down` recebe inicialização normal e `up` começa em zero: no primeiro
        # passo o termo aditivo é nulo e a rede reproduz exatamente o modelo
        # original, o que torna o treino monotônico a partir do baseline.
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scaling


class LoRALinear(nn.Module):
    """Equivalente de `LoRAConv2d` para camadas lineares."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scaling


def _layer_width(module: nn.Module) -> int:
    """Maior dimensão de canal da camada, usada como critério de tamanho."""
    if isinstance(module, nn.Linear):
        return max(module.in_features, module.out_features)
    return max(module.in_channels, module.out_channels)


def _is_adaptable(module: nn.Module, min_channels: int) -> bool:
    """Decide se vale decompor esta camada.

    Convoluções agrupadas — em particular as depthwise, com `groups` igual ao
    número de canais — já têm posto reduzido por construção, e decompô-las
    custaria mais parâmetros do que economizaria. Camadas estreitas estão no
    mesmo caso, por isso o piso de largura.
    """
    if isinstance(module, nn.Conv2d) and module.groups != 1:
        return False
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return False
    return _layer_width(module) >= min_channels


def inject_lora(model: nn.Module,
                rank: int = 16,
                alpha: float | None = None,
                include: tuple[str, ...] = ('backbone', 'neck'),
                min_channels: int = 32) -> dict[str, int]:
    """Substitui camadas elegíveis por versões adaptadas, no lugar.

    Args:
        model: modelo já construído e com pesos carregados.
        rank: posto da decomposição.
        alpha: escala do termo aditivo; o padrão iguala `rank`, resultando em
            fator unitário.
        include: prefixos dos submódulos a adaptar. O padrão cobre backbone e
            neck, que é onde a seletividade cromática reside; a cabeça de
            regressão é treinada integralmente e não precisa de adaptador.
        min_channels: camadas mais estreitas que isto são deixadas intactas,
            porque nelas a decomposição não reduz parâmetros de forma relevante.

    Returns:
        Contagem de camadas adaptadas por prefixo.
    """
    alpha = float(rank if alpha is None else alpha)
    adapted: dict[str, int] = {prefix: 0 for prefix in include}

    for prefix in include:
        parent_root = getattr(model, prefix, None)
        if parent_root is None:
            continue

        # Coletar antes de substituir: modificar a árvore durante a travessia
        # faria o iterador visitar os módulos recém-inseridos.
        targets = [(name, module)
                   for name, module in parent_root.named_modules()
                   if _is_adaptable(module, min_channels)]

        for name, module in targets:
            parent = parent_root
            *path, attribute = name.split('.')
            for step in path:
                parent = getattr(parent, step)
            wrapper = (LoRALinear if isinstance(module, nn.Linear) else LoRAConv2d)
            setattr(parent, attribute, wrapper(module, rank, alpha))
            adapted[prefix] += 1

    return adapted


def _stay_in_eval(self, mode: bool = True):
    """Substitui `train()` numa instância, para que ela ignore o modo."""
    return self


def freeze_normalization(module: nn.Module) -> int:
    """Mantém as camadas de normalização em modo de avaliação permanentemente.

    Marcar os parâmetros como `requires_grad=False` congela apenas os termos
    aprendíveis da BatchNorm. Em modo de treino ela continua fazendo duas coisas
    que quebram um backbone supostamente congelado: normaliza pelas estatísticas
    do lote em vez das acumuladas, e sobrescreve as acumuladas com elas. Num
    lote de doze amostras sob augmentation forte essas estatísticas são ruidosas
    e distantes das que calibraram o modelo — medido, a saída da mesma entrada
    diverge em até 8,4 entre os modos de avaliação e de treino.

    Chamar `.eval()` não basta, porque o MMEngine invoca `model.train()` a cada
    época e a chamada recorre nos filhos. Sobrescrever `train` na instância faz
    a camada ignorar a mudança de modo de forma duradoura.

    Returns:
        Quantidade de camadas de normalização congeladas.
    """
    frozen = 0
    for submodule in module.modules():
        if isinstance(submodule, _BatchNorm):
            submodule.eval()
            submodule.train = _stay_in_eval.__get__(submodule, type(submodule))
            frozen += 1
    return frozen


def freeze_except_lora(model: nn.Module,
                       trainable_prefixes: tuple[str, ...] = ('head', )) -> int:
    """Congela tudo, exceto os adaptadores e os prefixos indicados.

    Returns:
        Quantidade de camadas de normalização congeladas nos módulos que
        permanecem fixos.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            module.down.weight.requires_grad = True
            module.up.weight.requires_grad = True

    trainable_modules = []
    for prefix in trainable_prefixes:
        submodule = getattr(model, prefix, None)
        if submodule is not None:
            trainable_modules.append(submodule)
            for parameter in submodule.parameters():
                parameter.requires_grad = True

    # Congelar a normalização apenas onde os pesos ficam fixos. Nos módulos
    # treináveis a BatchNorm deve seguir se ajustando normalmente.
    frozen = 0
    for name, child in model.named_children():
        if child not in trainable_modules:
            frozen += freeze_normalization(child)
    return frozen


def adapter_output_magnitude(model: nn.Module) -> float:
    """Soma dos valores absolutos das projeções de saída dos adaptadores.

    Vale exatamente zero enquanto os adaptadores mantêm a inicialização
    original, situação em que o modelo adaptado é idêntico ao modelo base.

    Serve como detector de reinicialização acidental, que é um erro silencioso e
    caro: `Runner.train()` do MMEngine chama `init_weights()` *depois* de o
    modelo ter sido construído e carregado, e sem intervenção sobrescreve estes
    pesos com valores aleatórios. O efeito medido foi a soma saltar de 0 para
    77.158 e a perda virar NaN na primeira iteração.
    """
    total = 0.0
    for module in model.modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            total += float(module.up.weight.detach().abs().sum())
    return total


def parameter_summary(model: nn.Module) -> dict[str, float]:
    """Total e treináveis, em milhões de parâmetros."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
        'trainable_pct': 100.0 * trainable / total if total else 0.0,
    }


def merge_lora_state_dict(state_dict: dict, alpha: float | None = None) -> dict:
    """Funde os adaptadores nos pesos base e devolve nomes de camada originais.

    Um checkpoint treinado com LoRA não carrega num modelo comum: a injeção
    renomeia `conv.weight` para `conv.base.weight` e acrescenta `conv.down` e
    `conv.up`. Como o MMEngine trata chave ausente como aviso e não como erro,
    avaliar esse checkpoint num config sem LoRA roda até o fim e reporta a
    métrica de um modelo aleatório. Esta função existe para que isso não seja
    possível: ela devolve um `state_dict` que um modelo comum carrega inteiro.

    A fusão é exata, não uma aproximação. Para uma convolução, o termo aditivo
    `up(down(x))` compõe-se num único núcleo, porque `up` é ponto-a-ponto e
    `down` compartilha núcleo, passo e dilatação com a camada base; convoluções
    agrupadas nunca são adaptadas, justamente por não admitirem essa composição.

    Args:
        state_dict: pesos do checkpoint treinado com adaptadores.
        alpha: escala usada na injeção. `None` assume `alpha == rank`, que é o
            padrão de `inject_lora` e resulta em fator unitário.

    Returns:
        Novo dicionário, com os adaptadores fundidos e as chaves originais.
    """
    merged = {}
    adapters = 0

    for key, tensor in state_dict.items():
        if key.endswith('.down.weight') or key.endswith('.up.weight'):
            continue
        if '.base.' not in key:
            merged[key] = tensor
            continue

        prefix, _, suffix = key.rpartition('.base.')
        if suffix != 'weight':  # bias da camada base: só renomeia
            merged[f'{prefix}.{suffix}'] = tensor
            continue

        down = state_dict[f'{prefix}.down.weight']
        up = state_dict[f'{prefix}.up.weight']
        rank = down.shape[0]
        scaling = (rank if alpha is None else alpha) / rank

        if tensor.dim() == 4:
            delta = torch.einsum('or,rikl->oikl', up[:, :, 0, 0], down)
        else:
            delta = up @ down

        merged[f'{prefix}.weight'] = tensor + delta * scaling
        adapters += 1

    if adapters == 0:
        raise ValueError('nenhum adaptador encontrado: este checkpoint não foi '
                         'treinado com LoRA')
    return merged


def has_lora_adapters(state_dict: dict) -> bool:
    """Reconhece um checkpoint treinado com adaptadores pelo nome das chaves."""
    return any(
        key.endswith('.up.weight')
        and key.removesuffix('up.weight') + 'down.weight' in state_dict
        for key in state_dict)
