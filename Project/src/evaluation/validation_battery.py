"""Os testes de validação do Projeto Físico, executáveis.

Camada Model. O documento especifica sete testes de caixa-preta e caixa-branca,
e até aqui os números que os sustentavam viviam espalhados por medições avulsas,
cada uma com sua condição. Aqui cada teste vira função com critério de aceite
explícito, de modo que a tabela do capítulo de resultados saia de uma execução e
não de uma compilação manual.

Três deles --- precisão full-body, erro corporal no domínio veicular e
calibração do detector --- já tinham medição própria e são **citados**, com o
arquivo de origem declarado, em vez de reexecutados: rodar a avaliação completa
do COCO a cada bateria custaria meia hora para reproduzir um número que não
mudou. Os demais são medidos aqui.
"""

from __future__ import annotations

import time

import numpy as np

MILLIMETERS = 1000.0

# Orçamento do projeto: 20 FPS e 100ms de latência por quadro.
FPS_MINIMO = 20.0
LATENCIA_MAXIMA_MS = 100.0

# Critério do teste de oclusão, como especificado no documento.
DEGRADACAO_MAXIMA = 0.15

# Crescimento de memória tolerado numa execução longa, em MiB. Acima disso há
# retenção acumulando, que numa demonstração de minutos termina em falta de
# memória.
CRESCIMENTO_MAXIMO_MIB = 64.0


def latencia_por_quadro(pipeline, quadros, lifter=None,
                        sincroniza=None) -> dict[str, float]:
    """Teste 2: latência quadro a quadro sobre uma sequência real.

    Mede o caminho completo, e não o estimador isolado: detector, pose e, quando
    dado, o lifting. A sincronização da GPU é obrigatória --- sem ela, mede-se o
    custo de enfileirar trabalho, não o de executá-lo, e o número sai de três a
    quatro vezes otimista.
    """
    latencias = []
    for quadro in quadros:
        inicio = time.perf_counter()
        resultado = pipeline(quadro)
        if lifter is not None and resultado.num_people:
            altura, largura = quadro.shape[:2]
            lifter(resultado.keypoints[0], resultado.scores[0],
                   (largura, altura))
        if sincroniza is not None:
            sincroniza()
        latencias.append((time.perf_counter() - inicio) * 1e3)

    latencias = np.array(latencias)
    return {
        'quadros': len(latencias),
        'latencia_mediana_ms': round(float(np.median(latencias)), 2),
        'latencia_p90_ms': round(float(np.percentile(latencias, 90)), 2),
        'latencia_p99_ms': round(float(np.percentile(latencias, 99)), 2),
        'fps_mediano': round(float(1e3 / np.median(latencias)), 1),
        'aprovado': bool(1e3 / np.median(latencias) >= FPS_MINIMO
                         and np.percentile(latencias, 90) < LATENCIA_MAXIMA_MS),
    }


def ocluir(quadro: np.ndarray, centro: np.ndarray, lado: int) -> np.ndarray:
    """Apaga um quadrado em torno de um ponto, simulando oclusão por objeto."""
    ocluido = quadro.copy()
    x, y = int(centro[0]), int(centro[1])
    meio = lado // 2
    altura, largura = quadro.shape[:2]
    ocluido[max(0, y - meio):min(altura, y + meio),
            max(0, x - meio):min(largura, x + meio)] = 0
    return ocluido


def coordenadas_no_quadro(resultado, frame_size: tuple[int, int],
                          min_score: float) -> dict:
    """Teste 5: os keypoints saem no sistema do quadro, não no do recorte.

    O defeito que este teste pega é silencioso e já apareceu neste projeto: o
    `inference_topdown` devolve os keypoints já remapeados, e recortar à mão
    antes dele aplica a transformação duas vezes, deslocando tudo para o canto
    superior esquerdo sem levantar erro algum.

    Só entram os keypoints **observados**. A primeira versão deste teste
    contava os 133 e reprovava o sistema por causa das pernas extrapoladas, que
    estão fora do quadro de propósito --- o critério é que estava errado, não a
    saída.
    """
    largura, altura = frame_size
    pontos = resultado.keypoints[0]
    observados = resultado.scores[0] >= min_score
    dentro = ((pontos[:, 0] >= 0) & (pontos[:, 0] <= largura)
              & (pontos[:, 1] >= 0) & (pontos[:, 1] <= altura))
    fracao = float(dentro[observados].mean()) if observados.any() else 0.0

    # A caixa do detector é a referência independente: um keypoint observado
    # tem de cair perto dela, e não no canto da imagem.
    caixa = resultado.boxes[0]
    centro_caixa = np.array([(caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2])
    diagonal = float(np.linalg.norm(caixa[2:] - caixa[:2]))
    distancia = np.linalg.norm(pontos[observados] - centro_caixa, axis=-1) / diagonal
    return {
        'observados': int(observados.sum()),
        'fracao_no_quadro': round(fracao, 4),
        'distancia_mediana_ao_centro_da_caixa': round(float(np.median(distancia)), 3),
        'aprovado': bool(fracao > 0.95 and np.median(distancia) < 0.5),
    }


def fusao_temporal(lifter, keypoints, scores, frame_size) -> dict[str, float]:
    """Teste 6: a janela temporal reduz tremor comparada com quadro único.

    O quadro único é emulado alimentando a janela com dezesseis cópias do mesmo
    quadro --- a rede vê a mesma arquitetura e o mesmo peso, sem contexto
    temporal algum, que é exatamente a comparação que interessa.
    """
    from src.evaluation.live_quality import jitter_mm

    lifter.reset()
    temporal = np.stack([lifter(k, s, frame_size)
                         for k, s in zip(keypoints, scores)])

    unico = []
    for k, s in zip(keypoints, scores):
        lifter.reset()
        for _ in range(16):                    # janela cheia do mesmo quadro
            pose = lifter(k, s, frame_size)
        unico.append(pose)
    unico = np.stack(unico)

    tremor_temporal = jitter_mm(temporal)
    tremor_unico = jitter_mm(unico)
    corpo_t = tremor_temporal['ombros'] + tremor_temporal['cotovelos e pulsos']
    corpo_u = tremor_unico['ombros'] + tremor_unico['cotovelos e pulsos']
    return {
        'tremor_com_janela': {k: round(v, 2) for k, v in tremor_temporal.items()},
        'tremor_quadro_unico': {k: round(v, 2) for k, v in tremor_unico.items()},
        'reducao': round(float(1 - corpo_t / corpo_u), 4) if corpo_u else None,
        'aprovado': bool(corpo_t < corpo_u),
    }


def memoria_estavel(amostras_mib: list[float]) -> dict:
    """Teste 7: memória de GPU não cresce ao longo de uma execução longa."""
    inicio = float(np.mean(amostras_mib[:3]))
    fim = float(np.mean(amostras_mib[-3:]))
    return {
        'memoria_inicial_mib': round(inicio, 1),
        'memoria_final_mib': round(fim, 1),
        'crescimento_mib': round(fim - inicio, 1),
        'aprovado': bool(fim - inicio < CRESCIMENTO_MAXIMO_MIB),
    }
