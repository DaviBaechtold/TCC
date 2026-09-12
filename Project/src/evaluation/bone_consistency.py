"""Mede a qualidade de uma pose 3D sem precisar de referência alguma.

Camada Model. Existe porque a régua disponível não serve: a pose tridimensional
do Drive&Act, única referência do domínio de aplicação, tem ao menos 30,1mm de
incerteza por quadro --- mais que o dobro da diferença entre os modelos que se
quer comparar. Medir contra ela é medir o ruído dela.

O princípio aqui dispensa referência: **o comprimento de um osso é constante**.
A distância entre ombro e cotovelo da mesma pessoa não muda enquanto ela se
move, de modo que toda variação nela, na pose *predita*, é erro do modelo. Um
estimador perfeito produziria desvio zero.

O que a métrica capta e o que não capta, para que não seja superinterpretada:

**Capta** incoerência temporal e tremor --- a pose que oscila entre quadros, que
é exatamente o defeito que a fusão temporal do Módulo 3 existe para suprimir.

**Não capta** erro sistemático. Um modelo que encolha a pessoa inteira em 20%,
de forma consistente, tem desvio zero e está errado. A métrica é necessária e
não suficiente, e por isso acompanha o MPJPE em vez de substituí-lo.

Serve também de piso de comparação: a própria referência do Drive&Act mede
30,1mm nesta métrica, de modo que um modelo abaixo disso é mais coerente que a
anotação contra a qual está sendo avaliado.
"""

from __future__ import annotations

import numpy as np

MILLIMETERS = 1000.0

# Ossos cujos dois extremos a vista de retrovisor observa. Um osso com uma ponta
# extrapolada mediria a extrapolação, e não a coerência da pose.
OBSERVABLE_BONES = {
    'ombro a ombro': (5, 6),
    'braço esquerdo': (5, 7),
    'braço direito': (6, 8),
    'antebraço esquerdo': (7, 9),
    'antebraço direito': (8, 10),
    'quadril a quadril': (11, 12),
    'tronco esquerdo': (5, 11),
    'tronco direito': (6, 12),
}

# Abaixo disto a sequência é curta demais para que um desvio signifique algo.
MIN_FRAMES = 20


def bone_lengths(sequence: np.ndarray,
                 bones: dict[str, tuple[int, int]] = OBSERVABLE_BONES
                 ) -> dict[str, np.ndarray]:
    """Comprimento de cada osso, quadro a quadro.

    Args:
        sequence: [T, K, 3] em metros.

    Returns:
        Um vetor de T comprimentos por osso.
    """
    return {
        nome: np.linalg.norm(sequence[:, a] - sequence[:, b], axis=-1)
        for nome, (a, b) in bones.items()
    }


# Osso de referência para a versão invariante a escala. O tronco é o mais
# estável do corpo e o mais confiável na vista de retrovisor.
SCALE_REFERENCE = ('tronco esquerdo', 'tronco direito')


def consistency(sequence: np.ndarray,
                bones: dict[str, tuple[int, int]] = OBSERVABLE_BONES
                ) -> dict[str, float]:
    """Desvio do comprimento de cada osso ao longo da sequência, em milímetros.

    Mede duas coisas de uma vez, e a distinção importa:

    **Absoluto** (`mediana`) capta incoerência temporal **e** deriva de escala.
    O lifting monocular não conhece a escala real, de modo que o tamanho
    aparente da pessoa --- que muda quando ela se inclina para frente ou para
    trás --- se traduz em tamanho predito. Parte do desvio absoluto é, portanto,
    ambiguidade legítima da tarefa, e não erro.

    **Relativo** (`mediana_relativa`) divide cada osso pelo comprimento do
    tronco no mesmo quadro, e com isso mede só a coerência de **forma**. É a
    mesma lógica que leva o PA-MPJPE a alinhar antes de comparar: separa o que o
    método não se propõe a resolver do que ele deveria resolver.

    Returns:
        Desvio por osso em milímetros, `mediana`, e `mediana_relativa` em
        proporção do tronco. Vazio se a sequência for curta demais.
    """
    if len(sequence) < MIN_FRAMES:
        return {}

    comprimentos = bone_lengths(sequence, bones)
    desvios = {nome: float(np.std(valores) * MILLIMETERS)
               for nome, valores in comprimentos.items()}
    desvios['mediana'] = float(np.median(list(desvios.values())))

    escala = np.mean([comprimentos[nome] for nome in SCALE_REFERENCE
                      if nome in comprimentos], axis=0)
    if escala.ndim and np.all(escala > 1e-6):
        relativos = [float(np.std(valores / escala))
                     for nome, valores in comprimentos.items()
                     if nome not in SCALE_REFERENCE]
        if relativos:
            desvios['mediana_relativa'] = float(np.median(relativos))

    return desvios


def motion(sequence: np.ndarray) -> float:
    """Deslocamento mediano de um keypoint entre quadros vizinhos, em mm.

    Controle obrigatório da coerência de osso, e não um número de interesse
    próprio. A coerência tem um ponto cego grave: **uma pose congelada tem
    coerência perfeita**. Um modelo que aprenda a devolver sempre a pose média,
    ignorando a entrada, pontuaria melhor que um estimador correto.

    Comparar dois modelos pela coerência só é válido se ambos se moverem de
    forma parecida. Se o mais coerente também for o mais parado, o ganho é
    suavização disfarçada.
    """
    if len(sequence) < 2:
        return 0.0
    deslocamento = np.linalg.norm(np.diff(sequence, axis=0), axis=-1)
    return float(np.median(deslocamento) * MILLIMETERS)
