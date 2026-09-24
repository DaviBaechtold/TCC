"""Simula um estimador 2D imperfeito na entrada do treino do lifting.

Camada Model. Substitui `KeypointDropout`, que apagava keypoints e **piorou** o
domínio veicular: de 82,88mm para 108,37mm sob o mesmo estimador 2D. O motivo do
fracasso é instrutivo e está registrado aqui para não se repetir.

**O erro de raciocínio.** Zerar as pernas no H3WB dava 80,75mm, e o domínio real
dava 82,88mm; disso concluí que a corrupção real *era* zeros. Não é. Coincidiram
as magnitudes, não os mecanismos. Medindo onde o estimador 2D de fato coloca uma
junta que não está na imagem, sobre 200 quadros do Drive&Act, em fração da caixa
da pessoa:

    joelho esquerdo   x 0,91 ± 0,09   y 0,92 ± 0,12
    tornozelo direito x 0,34 ± 0,15   y 1,07 ± 0,15
    dedão direito     x 0,40 ± 0,12   y 1,07 ± 0,13

Ele não espalha nada: coloca a junta **junto da borda inferior do recorte**, onde
a perna estaria se continuasse fora de quadro. É uma extrapolação plausível e
consistente, não ruído. Treinar contra zeros ensinou a rede a reconhecer um sinal
que nunca ocorre.

**O que a rede precisa aprender.** O terceiro canal da entrada carrega a
confiança, e no H3WB ele é **constante em exatamente 1,0** --- desvio padrão
zero, um único valor distinto em todo o conjunto. A rede aprendeu a ignorá-lo, e
a medição confirma: alimentar o canal com 1,0, com a resposta bruta ou com ela
normalizada muda o erro em menos de 2,5%. No habitáculo esse canal é justamente
o que separa junta observada de junta extrapolada --- a resposta média é 8,30
contra 5,13 --- e a rede é surda a ele.

Esta transformação torna o canal informativo: quem recebe confiança baixa recebe
também posição deslocada, de modo que a correlação entre os dois exista no treino
e possa ser aprendida.

**O corte de quadro (v3).** Os grupos anatômicos acima cobrem o membro que sai
de quadro, mas nunca o quadril: `KEYPOINT_GROUPS` não contém os índices 11 e 12
em grupo algum, de modo que o lifting v2 jamais viu um quadril escondido. É
exatamente o que a webcam de mesa entrega. Medido na gravação do usuário, a
Etapa 2 coloca os dois quadris em y mediano 706 e 708 num quadro de 720 px --- 13
px acima da borda inferior --- com resposta 6,1 de 8,3, que passa o limiar de
detecção em 95% dos quadros. Sob esse corte aplicado ao H3WB com verdade de
campo, o v2 erra 507mm nos quadris: o tronco colapsa e os quadris sobem à altura
dos ombros. Devolvendo os quadris verdadeiros, as pernas caem de 239mm para
180mm --- o erro das pernas é consequência do quadril perdido, não das pernas.

O corte modela isso como o que de fato é: **uma linha horizontal na imagem**,
abaixo da qual tudo é extrapolado até a borda, qualquer que seja a anatomia da
junta. A câmera é estática, então a linha é uma só por janela; o corpo se move,
então a junta pode cruzá-la dentro da janela, e a pertinência é avaliada quadro a
quadro. Isso não é a "ausência intermitente" que o sorteio por quadro produziria
e que o projeto rejeita: a linha não muda, quem muda de lado é o corpo.
"""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

# Grupos anatômicos que somem juntos numa cena real: um membro sai de quadro
# inteiro, não meia canela.
KEYPOINT_GROUPS = {
    'pernas_e_pes': list(range(13, 23)),
    'perna_esquerda': [13, 15, 17, 18, 19],
    'perna_direita': [14, 16, 20, 21, 22],
    'pes': list(range(17, 23)),
    'braco_esquerdo': [7, 9],
    'braco_direito': [8, 10],
    'mao_esquerda': list(range(91, 112)),
    'mao_direita': list(range(112, 133)),
    'face': list(range(23, 91)),
}

# A vista de retrovisor perde pernas e pés em todo quadro, e é o caso de
# aplicação; por isso ele é sorteado com peso maior.
GROUP_WEIGHTS = {
    'pernas_e_pes': 4.0, 'perna_esquerda': 1.0, 'perna_direita': 1.0,
    'pes': 2.0, 'braco_esquerdo': 1.0, 'braco_direito': 1.0,
    'mao_esquerda': 1.0, 'mao_direita': 1.0, 'face': 1.0,
}

# Onde a junta extrapolada aparece, em fração do envelope do corpo visível.
# Medido no Drive&Act: y junto da borda inferior, x espalhado pela largura.
EXTRAPOLATED_Y = (0.88, 1.12)
EXTRAPOLATED_X = (0.25, 0.95)
EXTRAPOLATED_JITTER = 0.13

# Confiança em [0, 1], obtida dividindo a resposta medida pela média dos
# keypoints observáveis (8,30). Observáveis ficam perto de 1; extrapolados, de
# 0,62, com a dispersão medida entre os percentis 10 e 90.
CONFIDENCE_OBSERVED = (0.85, 1.00)
CONFIDENCE_EXTRAPOLATED = (0.37, 0.92)

# Teto de confiança para a junta que o sistema **sabe** não ter observado.
# Contrato compartilhado: o painel aplica o mesmo teto em inferência, em
# `src/models/observability.py`, e o treino precisa ter visto essa faixa.
#
# O valor vem do limiar de detecção do estimador: 3,0 de resposta bruta sobre a
# média 8,30 dos observáveis dá 0,36. Abaixo desse ponto a junta seria
# descartada, de modo que 0,3 é o maior valor que ainda significa
# inequivocamente "isto não foi visto", e fica abaixo dos 0,37 onde a faixa de
# extrapolado começa --- uma faixa exclusiva que a rede pode aprender.
UNOBSERVED_CONFIDENCE_CAP = 0.3

# Fração das juntas cortadas que mesmo assim chega com a confiança alta de quem
# parece observado. O critério de borda do painel erra para menos: com margem de
# 28 px ele marca 99,3% dos quadris, ou seja perde 0,7% deles. Esse é o piso, não
# a taxa: o critério só enxerga a borda da imagem, e um corte produzido por um
# obstáculo *dentro* do quadro --- a quina da mesa acima da borda --- não dispara
# nada. Um quinto é a margem escolhida para que a rede não aprenda "confiança
# baixa" como condição necessária do corte; não é uma medição.
CUT_LOOKS_OBSERVED_PROB = 0.20

# Onde a linha de corte cai, em fração do vão entre a linha dos ombros e a dos
# joelhos. O intervalo cobre as duas montagens sem precisar distingui-las: num
# adulto o quadril fica perto de 0,45 desse vão, de modo que um sorteio uniforme
# dá aproximadamente metade de cortes de mesa (acima do quadril) e metade de
# cortes de retrovisor (entre quadril e joelho). O piso em 0,10 existe porque
# abaixo dele a própria linha dos ombros, que define o corte, seria cortada.
CUT_LEVEL_RANGE = (0.10, 1.00)

# Quanto a junta extrapolada para antes da linha de corte, em larguras de ombro.
# Medido na gravação: quadris em y mediano 706 e 708 num quadro de 720 px, ou
# seja 13 px acima da borda, contra uma largura de ombros de 223 px.
CUT_INSET_SHOULDER_WIDTHS = 13.0 / 223.0

# Tremor por quadro da junta presa à borda, em larguras de ombro. A gravação dá
# 7 px de segunda diferença quadro a quadro; para ruído independente entre
# quadros a segunda diferença tem desvio sqrt(6) vezes o da posição, o que
# devolve 2,9 px, ou 0,013 largura de ombro. É redesenhado a cada quadro --- ao
# contrário do deslocamento por grupo, que é fixo na janela --- porque a borda
# não segura a junta parada, ela a segura oscilando.
CUT_JITTER_SHOULDER_WIDTHS = 2.9 / 223.0

# Índices COCO-WholeBody das juntas que definem a geometria do corte.
SHOULDER_INDICES = [5, 6]
KNEE_INDICES = [13, 14]

# Como o estimador coloca uma junta cortada, medido em
# `scripts/measure_absent_placement.py` sobre 290 quadros da webcam e 300 do
# Drive&Act (`results/posicao_ausentes_*.json`). A primeira versão do corte
# prendia **tudo** na linha, e por isso corrigiu o quadril e não as pernas.
#
# O que a medição mostra é que a colocação depende da distância da junta à
# linha, não da anatomia dela. A junta imediatamente abaixo do corte --- o
# quadril, na webcam de mesa --- é grudada na borda em 98 e 99% dos quadros. As
# mais profundas escapam: joelho em 43 a 53%, tornozelo em 31 a 59%, pés em 6 a
# 40%. O que não encosta na borda cai **sobre o corpo visível** em 17 a 44% dos
# casos --- joelho no peito, tornozelo ao lado da mão levantada --- ou se
# espalha, chegando acima da linha dos ombros (p10 de -1,3 largura de ombro no
# dedinho).
# A probabilidade de encostar na borda **decai com a distância à linha**, e não
# tem degrau: um limiar rígido classificava o quadril como junta profunda e o
# prendia em 44% dos casos contra os 98% medidos. A forma gaussiana é escolha de
# conveniência, ancorada em três pontos da medição --- quadril a ~0,4 largura de
# ombro abaixo da linha prende 0,98, joelho a ~1,2 prende 0,43 a 0,53, tornozelo
# e pés a ~2,2 prendem 0,06 a 0,49.
CUT_PIN_PROB_NEAR = 0.95
CUT_PIN_PROB_FAR = 0.25
CUT_PIN_DECAY_SHOULDERS = 0.9

# Do que não encosta na borda, a parcela que gruda num keypoint visível ---
# joelho no peito, tornozelo ao lado da mão levantada. Medido: 17 a 44%.
CUT_ON_BODY_PROB = 0.30

# Onde cai o que nem encosta nem gruda, em larguras de ombro a partir da linha.
# A faixa vem dos percentis 10 e 90 medidos, e inclui posições acima da linha
# dos ombros --- o dedinho esquerdo chega a -0,9.
CUT_SCATTER_Y_SHOULDERS = (-1.0, 1.5)


@TRANSFORMS.register_module()
class SimulatedEstimatorNoise(BaseTransform):
    """Corrompe a entrada 2D como o estimador real corrompe, e rebaixa a
    confiança junto.

    Dois modos, exclusivos entre si: o grupo anatômico, que apaga um membro
    inteiro, e o corte de quadro, que apaga tudo abaixo de uma linha horizontal.

    Args:
        prob: probabilidade de aplicar o modo de grupo anatômico a um exemplo.
        max_groups: quantos grupos podem ser afetados ao mesmo tempo.
        frame_cut_prob: probabilidade de aplicar o corte de quadro. Os dois modos
            são exclusivos: o deslocamento por grupo mede o envelope do corpo
            visível, e medi-lo sobre um corpo já truncado empilharia dois
            deslocamentos que nunca coexistem --- a câmera tem um enquadramento
            só.
        unobserved_confidence: teto de confiança das juntas deslocadas, para
            casar com o teto que o painel aplica em inferência. Com `None` vale a
            faixa de extrapolado medida no Drive&Act, que é o comportamento do v2.
    """

    def __init__(self, prob: float = 0.6, max_groups: int = 2,
                 frame_cut_prob: float = 0.0,
                 unobserved_confidence: float | None = None,
                 cut_placement: str = 'linha') -> None:
        super().__init__()
        self.prob = prob
        self.max_groups = max_groups
        self.frame_cut_prob = frame_cut_prob
        self.unobserved_confidence = unobserved_confidence
        if cut_placement not in ('linha', 'medido'):
            raise ValueError(f'colocação desconhecida: {cut_placement}')
        # `linha` prende tudo na linha de corte e é o que treinou o v3; fica
        # como padrão para que aquele config continue reproduzível. `medido`
        # usa a mistura de três modos que a medição descreve.
        self.cut_placement = cut_placement
        self._names = list(KEYPOINT_GROUPS)
        weights = np.array([GROUP_WEIGHTS[n] for n in self._names], float)
        self._probabilities = weights / weights.sum()

    def transform(self, results: dict) -> dict:
        labels = results.get('keypoint_labels')
        if labels is None:
            return results

        # Entrada que já chega corrompida pelo estimador real não é simulada de
        # novo: o Drive&Act traz as pernas grudadas na borda de verdade, e
        # empilhar a simulação por cima ensinaria uma corrupção dupla que não
        # ocorre. Ver `src/data/driveact_lift_dataset.py`.
        if results.get('corrupcao_real'):
            return results

        labels = labels.copy()

        # A confiança dos keypoints preservados também varia, ainda que pouco.
        # Sem isso o canal continuaria quase constante e a rede não teria motivo
        # para consultá-lo.
        labels[..., 2] = np.random.uniform(*CONFIDENCE_OBSERVED,
                                           size=labels.shape[:-1])

        # O curto-circuito é deliberado: com `frame_cut_prob` em zero nenhum
        # número aleatório é consumido aqui, e a sequência do gerador continua
        # idêntica à do v2, cujo config precisa seguir reprodutível.
        # `tests/test_estimator_noise.py` trava isso contra o HEAD do git.
        if self.frame_cut_prob > 0.0 and np.random.rand() < self.frame_cut_prob:
            return self.apply_frame_cut(results, labels)[0]

        if np.random.rand() >= self.prob:
            results['keypoint_labels'] = labels
            return results

        count = np.random.randint(1, self.max_groups + 1)
        chosen = np.random.choice(self._names, size=count, replace=False,
                                  p=self._probabilities)
        affected = sorted({index for name in chosen
                           for index in KEYPOINT_GROUPS[name]})

        retained = [k for k in range(labels.shape[-2]) if k not in affected]
        if not retained:
            results['keypoint_labels'] = labels
            return results

        self.extrapolate(labels, affected, retained)
        results['keypoint_labels'] = labels

        if 'keypoint_labels_visible' in results:
            visible = results['keypoint_labels_visible'].copy()
            visible[..., affected] = 0.0
            results['keypoint_labels_visible'] = visible

        return results

    def apply_frame_cut(self, results: dict, labels: np.ndarray,
                        level: float | None = None
                        ) -> tuple[dict, np.ndarray | None]:
        """Prende à linha de corte tudo o que estiver abaixo dela.

        Pública, e não privada, porque `src/evaluation/truncation_protocol.py`
        precisa cortar exatamente como o treino corta. Reimplementar a
        colocação do outro lado mediria um mecanismo diferente do treinado,
        que é o erro de inferência que já custou 25mm a este projeto.

        Args:
            level: altura da linha, em fração do vão ombro-joelho. `None`
                sorteia, que é o regime de treino; um valor fixo torna a
                avaliação reproduzível.

        Returns:
            `results` corrompido e a máscara `(T, K)` do que ficou abaixo da
            linha --- `None` quando a janela não sustenta corte algum.

        A linha é horizontal e única na janela, porque a câmera é estática. A
        pertinência é por quadro, porque o corpo não é: uma mão que desce abaixo
        da quina da mesa some naquele quadro e reaparece no seguinte, e é isso
        que a gravação mostra.
        """
        cut = self.cut_mask(labels, level)
        if cut is None:
            results['keypoint_labels'] = labels
            return results, None

        below, cut_y, body_scale = cut

        # x fica onde a junta estava, y encosta na linha: é o que o estimador
        # faz ao extrapolar um membro que continua fora de quadro.
        jitter = np.random.normal(
            0.0, CUT_JITTER_SHOULDER_WIDTHS * body_scale,
            size=labels.shape[:-1] + (2, ))
        pinned_y = cut_y - CUT_INSET_SHOULDER_WIDTHS * body_scale
        placed_x = labels[..., 0] + jitter[..., 0]
        placed_y = pinned_y + jitter[..., 1]

        if self.cut_placement == 'medido':
            placed_x, placed_y = self._place_as_measured(
                labels, below, cut_y, body_scale, placed_x, placed_y)

        labels[..., 0] = np.where(below, placed_x, labels[..., 0])
        labels[..., 1] = np.where(below, placed_y, labels[..., 1])
        labels[..., 2] = np.where(below,
                                  self._draw_low_confidence(labels.shape[:-1]),
                                  labels[..., 2])

        results['keypoint_labels'] = labels

        if 'keypoint_labels_visible' in results:
            visible = results['keypoint_labels_visible'].copy()
            visible[below] = 0.0
            results['keypoint_labels_visible'] = visible

        return results, below

    def _place_as_measured(self, labels: np.ndarray, below: np.ndarray,
                           cut_y: float, body_scale: float,
                           pinned_x: np.ndarray, pinned_y: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Sorteia o modo de colocação de cada junta cortada, como medido.

        Três modos, e a proporção entre eles depende de quão longe da linha a
        junta de fato estaria: a que está logo abaixo é grudada na borda quase
        sempre, a profunda escapa em mais da metade dos quadros. O modo é
        sorteado **uma vez por janela**, porque o enquadramento não muda dentro
        dela --- sortear por quadro ensinaria uma instabilidade que o estimador
        real não tem.
        """
        profundidade = np.maximum(
            (labels[..., 1].mean(axis=0) - cut_y) / body_scale, 0.0)
        probabilidade = CUT_PIN_PROB_FAR + (
            CUT_PIN_PROB_NEAR - CUT_PIN_PROB_FAR) * np.exp(
                -(profundidade / CUT_PIN_DECAY_SHOULDERS) ** 2)

        prende = np.random.rand(labels.shape[-2]) < probabilidade
        # Quem não encosta na borda ou gruda num keypoint visível, ou se
        # espalha pela faixa medida.
        no_corpo = ~prende & (np.random.rand(labels.shape[-2])
                              < CUT_ON_BODY_PROB)

        visiveis = np.flatnonzero(~below.any(axis=0))
        if visiveis.size:
            ancora = np.random.choice(visiveis, size=labels.shape[-2])
            corpo_x = labels[..., ancora, 0]
            corpo_y = labels[..., ancora, 1]
        else:                       # janela sem nada visível: nada a copiar
            no_corpo = np.zeros_like(no_corpo)
            corpo_x = corpo_y = pinned_x

        espalhado_y = cut_y + body_scale * np.random.uniform(
            *CUT_SCATTER_Y_SHOULDERS, size=labels.shape[-2])
        espalhado_x = labels[..., 0].mean(axis=0) + body_scale * np.random.normal(
            0.0, 0.5, size=labels.shape[-2])

        x = np.where(prende, pinned_x,
                     np.where(no_corpo, corpo_x, espalhado_x + 0.0 * pinned_x))
        y = np.where(prende, pinned_y,
                     np.where(no_corpo, corpo_y, espalhado_y + 0.0 * pinned_y))
        return x, y

    def cut_mask(self, labels: np.ndarray, level: float | None = None
                 ) -> tuple[np.ndarray, float, float] | None:
        """Sorteia a linha de corte e devolve quem cai abaixo dela.

        Returns:
            A máscara `(T, K)` de quem está abaixo, a altura da linha e a
            largura de ombros que serve de escala --- ou `None`.

        Devolve `None` quando a janela não tem geometria para sustentar um corte
        --- ombros e joelhos colados, ou pose invertida --- em vez de inventar
        uma linha arbitrária.
        """
        shoulders = labels[..., SHOULDER_INDICES, :2]
        knees = labels[..., KNEE_INDICES, :2]

        shoulder_y = float(np.median(shoulders[..., 1]))
        knee_y = float(np.median(knees[..., 1]))
        span = knee_y - shoulder_y

        # A largura de ombros é a régua da gravação (223 px), e é o que converte
        # os 13 px de recuo e os 2,9 px de tremor para a escala desta janela.
        # Num perfil quase puro ela encolhe, e recuo e tremor encolhem junto: a
        # junta para exatamente na linha, que é uma degradação inofensiva.
        body_scale = float(np.median(np.linalg.norm(
            shoulders[..., 0, :] - shoulders[..., 1, :], axis=-1)))
        if span <= 0.0 or body_scale <= 0.0:
            return None

        if level is None:
            level = np.random.uniform(*CUT_LEVEL_RANGE)
        cut_y = shoulder_y + level * span

        below = labels[..., 1] > cut_y
        if not below.any():
            return None
        return below, cut_y, body_scale

    def _draw_low_confidence(self, shape: tuple[int, ...]) -> np.ndarray:
        """Confiança das juntas deslocadas, com ou sem o teto do contrato.

        Sem teto vale a faixa medida no Drive&Act, que se sobrepõe à dos
        observados de propósito. Com teto, a maioria cai na faixa exclusiva que o
        painel produz ao reconhecer a junta como não observada, e a minoria de
        `CUT_LOOKS_OBSERVED_PROB` continua na faixa alta --- que é onde o quadril
        real de fato responde, 0,73 --- para o caso em que o critério de borda
        não dispara.
        """
        if self.unobserved_confidence is None:
            return np.random.uniform(*CONFIDENCE_EXTRAPOLATED, size=shape)

        looks_observed = np.random.rand(*shape) < CUT_LOOKS_OBSERVED_PROB
        return np.where(
            looks_observed,
            np.random.uniform(*CONFIDENCE_EXTRAPOLATED, size=shape),
            np.random.uniform(0.0, self.unobserved_confidence, size=shape))

    def extrapolate(self, labels: np.ndarray, affected: list[int],
                     retained: list[int]) -> None:
        """Move os keypoints afetados para a borda do corpo visível.

        O grupo é deslocado de forma consistente na janela inteira, e não quadro
        a quadro: um membro fora de quadro continua fora enquanto a câmera não se
        mexe. Sortear por quadro ensinaria a rede a esperar uma ausência
        intermitente, que não é o regime da aplicação.
        """
        visible = labels[..., retained, :2]
        lower = visible.min(axis=(-3, -2))
        upper = visible.max(axis=(-3, -2))
        extent = np.maximum(upper - lower, 1e-6)

        count = len(affected)
        fraction = np.stack([
            np.random.uniform(*EXTRAPOLATED_X, size=count),
            np.random.uniform(*EXTRAPOLATED_Y, size=count),
        ], axis=-1)
        placed = lower + fraction * extent
        placed += np.random.normal(0.0, EXTRAPOLATED_JITTER * extent.mean(),
                                   placed.shape)

        labels[..., affected, :2] = placed
        labels[..., affected, 2] = self._draw_low_confidence(
            labels.shape[:-2] + (count, ))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'max_groups={self.max_groups}, '
                f'frame_cut_prob={self.frame_cut_prob}, '
                f'unobserved_confidence={self.unobserved_confidence})')
