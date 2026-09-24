#!/usr/bin/env python
"""Verifica que a simulação do estimador 2D produz o sinal que ela promete.

A transformação anterior, que apagava keypoints, não tinha teste e piorou o
domínio veicular em 25mm. O que ela errava não era código: era a hipótese. Este
teste não valida a hipótese --- só a medição faz isso --- mas trava as
propriedades sem as quais a hipótese não chega a ser exercitada.

O corte de quadro do v3 acrescenta uma obrigação nova: o v2 precisa continuar
reprodutível. Por isso a verificação 6 compara, número a número, a saída desta
versão com a da última revisão anterior ao corte, incluindo quantos números
aleatórios cada uma consome.

Executar:  python tests/test_estimator_noise.py
"""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmpose.registry import TRANSFORMS

from src.data.estimator_noise import (CONFIDENCE_EXTRAPOLATED,
                                      CONFIDENCE_OBSERVED,
                                      CUT_INSET_SHOULDER_WIDTHS,
                                      CUT_JITTER_SHOULDER_WIDTHS,
                                      CUT_LOOKS_OBSERVED_PROB, KEYPOINT_GROUPS,
                                      KNEE_INDICES, SHOULDER_INDICES,
                                      UNOBSERVED_CONFIDENCE_CAP,
                                      SimulatedEstimatorNoise)

SEQUENCE, KEYPOINTS = 16, 133
# Relativo ao diretório passado em `git -C`, e não à raiz do repositório: o
# código vive numa pasta dentro de um repositório maior, e fixar o prefixo
# quebraria o teste se essa pasta mudasse de lugar. O `:./` no `git show` faz a
# mesma resolução relativa.
CAMINHO_NO_REPOSITORIO = 'src/data/estimator_noise.py'

HIPS = [11, 12]
WRISTS = [9, 10]

# Esqueleto sentado em coordenadas já normalizadas pelo codec do MotionBERT:
# x e y em [-1, 1] com y crescendo para baixo, que é o espaço em que esta
# transformação opera. Largura de ombros 0,16 e vão ombro-joelho 0,60, as
# proporções de um adulto a alguns metros da câmera.
LINHAS_DO_CORPO = {
    -0.30: list(range(0, 5)) + list(range(23, 91)),   # cabeça e face
    -0.20: SHOULDER_INDICES,
    -0.05: [7, 8],                                    # cotovelos
    0.08: WRISTS + list(range(91, 133)),              # pulsos e mãos
    0.10: HIPS,
    0.40: KNEE_INDICES,
    0.70: [15, 16],                                   # tornozelos
    0.75: list(range(17, 23)),                        # pés
}
LARGURA_DE_OMBROS = 0.16


def uma_janela() -> dict:
    """Janela com posições distintas por keypoint e confiança constante em 1.

    Confiança constante é como o H3WB de fato chega: desvio padrão zero, um
    único valor distinto em todo o conjunto. É o que a transformação existe
    para quebrar.
    """
    labels = np.zeros((SEQUENCE, KEYPOINTS, 3), dtype=np.float32)
    labels[..., 0] = np.linspace(-1, 1, KEYPOINTS)
    labels[..., 1] = np.linspace(-1, 1, KEYPOINTS)
    labels[..., 2] = 1.0
    return {'keypoint_labels': labels}


def uma_janela_sentada(deriva: float = 0.0) -> dict:
    """Janela com geometria de corpo sentado, que é o que o corte exige.

    `deriva` desloca o corpo inteiro para baixo ao longo da janela. Serve para
    separar duas coisas que a janela estática confunde: o corpo, que se move, e
    a linha de corte, que não se move.
    """
    labels = np.zeros((SEQUENCE, KEYPOINTS, 3), dtype=np.float32)
    for altura, indices in LINHAS_DO_CORPO.items():
        labels[:, indices, 1] = altura
    labels[:, SHOULDER_INDICES, 0] = [-LARGURA_DE_OMBROS / 2,
                                      LARGURA_DE_OMBROS / 2]
    labels[:, HIPS, 0] = [-0.05, 0.05]
    labels[:, KNEE_INDICES, 0] = [-0.07, 0.07]
    labels[..., 1] += np.linspace(0.0, deriva, SEQUENCE)[:, None]
    labels[..., 2] = 1.0
    return {'keypoint_labels': labels}


def um_resultado_com_alvos() -> dict:
    """Janela acompanhada do alvo 3D e dos pesos da perda, como no treino real."""
    results = uma_janela_sentada()
    results['keypoint_labels_visible'] = np.ones((SEQUENCE, KEYPOINTS),
                                                 dtype=np.float32)
    results['lifting_target_label'] = np.ones((SEQUENCE, KEYPOINTS, 3),
                                              dtype=np.float32)
    results['lifting_target_visible'] = np.ones((SEQUENCE, KEYPOINTS),
                                                dtype=np.float32)
    results['lifting_target_weight'] = np.ones((SEQUENCE, KEYPOINTS),
                                               dtype=np.float32)
    return results


def _versao_anterior_ao_corte() -> types.ModuleType:
    """Carrega a última revisão do arquivo que ainda não conhecia o corte.

    Fixar `HEAD` deixaria de guardar coisa alguma assim que o v3 fosse commitado
    --- o teste passaria a comparar o arquivo consigo mesmo. Procurar para trás
    pela última revisão sem `frame_cut_prob` mantém a comparação viva contra a
    versão que treinou o v2, que é a que precisa seguir reprodutível.
    """
    raiz = Path(__file__).resolve().parents[1]
    # `--follow` e o caminho de cada revisão: em 24/09/2026 a raiz do repositório
    # passou de TCC/ para TCC/Project/, e as revisões antigas guardam o arquivo
    # sob `Project/`. Sem isso o histórico anterior à mudança some da busca.
    saida = subprocess.check_output(
        ['git', '-C', str(raiz), 'log', '--follow', '--format=%H',
         '--name-only', '--', CAMINHO_NO_REPOSITORIO], text=True).split()
    revisoes = list(zip(saida[0::2], saida[1::2]))

    for revisao, caminho in revisoes:
        fonte = subprocess.check_output(
            ['git', '-C', str(raiz), 'show', f'{revisao}:{caminho}'],
            text=True)
        if 'frame_cut_prob' not in fonte:
            return _executar_como_modulo(fonte, revisao)

    raise AssertionError('nenhuma revisão anterior ao corte de quadro no git')


def _executar_como_modulo(fonte: str, revisao: str) -> types.ModuleType:
    """Executa o código antigo sem deixá-lo brigar com o atual no registro.

    As duas versões declaram `SimulatedEstimatorNoise`, e o registro do MMPose
    recusa o nome repetido. A entrada atual sai antes e volta depois, de modo
    que nada além deste teste enxerga a troca.
    """
    nome = 'SimulatedEstimatorNoise'
    atual = TRANSFORMS._module_dict.pop(nome, None)
    try:
        modulo = types.ModuleType(f'estimator_noise_{revisao[:8]}')
        exec(compile(fonte, f'<{revisao[:8]}:estimator_noise.py>', 'exec'),
             modulo.__dict__)
        return modulo
    finally:
        TRANSFORMS._module_dict.pop(nome, None)
        if atual is not None:
            TRANSFORMS._module_dict[nome] = atual


def _deslocamento(depois: np.ndarray, antes: np.ndarray) -> np.ndarray:
    return np.linalg.norm(depois[..., :2] - antes[..., :2], axis=-1)


def verifica_modo_de_grupo():
    """Verificações 1 a 5: o modo de grupo anatômico, que é o que o v2 treinou."""
    np.random.seed(0)

    # 1. A confiança deixa de ser constante mesmo quando nada é deslocado.
    intocada = SimulatedEstimatorNoise(prob=0.0).transform(uma_janela())
    confianca = intocada['keypoint_labels'][..., 2]
    assert confianca.std() > 0.01, 'a confiança continuou constante'
    assert CONFIDENCE_OBSERVED[0] <= confianca.min() <= confianca.max() <= 1.0
    print(f'  confiança varia mesmo sem deslocar    desvio {confianca.std():.3f}  OK')

    # 2. Confiança média mais baixa acompanha posição deslocada. A correlação
    #    é o ponto inteiro: sem ela a rede não tem por que consultar o canal.
    original = uma_janela()['keypoint_labels']
    ruidosa = SimulatedEstimatorNoise(prob=1.0).transform(uma_janela())
    labels = ruidosa['keypoint_labels']
    deslocamento = _deslocamento(labels, original)

    # O marcador de quem foi extrapolado é o deslocamento, e não a confiança:
    # as duas faixas se sobrepõem de propósito, como se vê na verificação 3.
    afetados = deslocamento.max(axis=0) > 1e-6
    assert afetados.any(), 'nenhum keypoint foi deslocado'
    confianca = labels[..., 2]
    assert confianca[:, afetados].mean() < confianca[:, ~afetados].mean(), (
        'a confiança média não distingue extrapolado de observado')
    print(f'  confiança acompanha deslocamento      '
          f'{confianca[:, afetados].mean():.2f} contra '
          f'{confianca[:, ~afetados].mean():.2f}  OK')

    # 3. As faixas se sobrepõem, e isso é fiel e não descuido. Na medição real,
    #    92% das juntas invisíveis passam o limiar de detecção. Simular uma
    #    separação mais limpa que a realidade produziria um modelo que confia
    #    demais no canal justamente onde ele é ambíguo.
    assert CONFIDENCE_EXTRAPOLATED[1] > CONFIDENCE_OBSERVED[0], (
        'as faixas deixaram de se sobrepor; a simulação ficou otimista demais')
    sobreposicao = CONFIDENCE_EXTRAPOLATED[1] - CONFIDENCE_OBSERVED[0]
    print(f'  faixas se sobrepõem, como no real     {sobreposicao:.2f}  OK')

    # 4. O grupo some da janela inteira. Um membro fora de quadro continua fora
    #    enquanto a câmera não se mexe; ausência intermitente é outro regime.
    assert np.allclose(deslocamento[:, afetados],
                       deslocamento[0, afetados]), (
        'o deslocamento variou entre quadros da mesma janela')
    print(f'  ausência consistente na janela        '
          f'{int(afetados.sum())} keypoints  OK')

    # 5. Com o corte desligado --- que é o padrão, e é o que o v2 usa --- os
    #    índices removidos ainda formam grupos anatômicos, e não juntas avulsas.
    transformacao = SimulatedEstimatorNoise(prob=1.0, frame_cut_prob=0.0)
    ruidosa = transformacao.transform(uma_janela())
    afetados = _deslocamento(ruidosa['keypoint_labels'],
                             original).max(axis=0) > 1e-6
    indices = set(np.flatnonzero(afetados).tolist())
    assert any(indices >= set(g) for g in KEYPOINT_GROUPS.values()), (
        'o conjunto removido não corresponde a nenhum grupo anatômico')
    print('  remoção por grupo anatômico                          OK')


def verifica_identidade_com_a_versao_do_v2():
    """Verificação 6: os padrões reproduzem o v2 bit a bit, sorteios inclusive."""
    anterior = _versao_anterior_ao_corte()

    for construtor in (uma_janela, uma_janela_sentada):
        for semente in range(200):
            np.random.seed(semente)
            antigo = anterior.SimulatedEstimatorNoise().transform(construtor())
            # O sorteio seguinte denuncia consumo diferente do gerador mesmo
            # quando a saída coincide por acaso.
            marca_antiga = np.random.rand()

            np.random.seed(semente)
            novo = SimulatedEstimatorNoise().transform(construtor())
            marca_nova = np.random.rand()

            assert np.array_equal(antigo['keypoint_labels'],
                                  novo['keypoint_labels']), (
                f'saída diferente da do v2 em {construtor.__name__}/{semente}')
            assert marca_antiga == marca_nova, (
                f'consumo do gerador diferente do v2 em {semente}')

    print('  padrões idênticos ao v2, 400 janelas                 OK')


def verifica_linha_de_corte():
    """Verificação 7: uma linha horizontal separa quem foi preso de quem não foi."""
    original = uma_janela_sentada()['keypoint_labels']
    np.random.seed(7)
    cortada = SimulatedEstimatorNoise(
        frame_cut_prob=1.0).transform(uma_janela_sentada())['keypoint_labels']

    movidos = _deslocamento(cortada, original)[0] > 0.0
    assert movidos.any() and not movidos.all()

    abaixo, acima = original[0, movidos, 1], original[0, ~movidos, 1]
    assert acima.max() < abaixo.min(), (
        'quem foi preso e quem não foi não são separáveis por uma linha')

    # Todos os presos param na mesma altura, alguns milésimos antes da linha.
    altura = cortada[:, movidos, 1]
    tolerancia = 4 * CUT_JITTER_SHOULDER_WIDTHS * LARGURA_DE_OMBROS
    assert altura.std() < tolerancia, 'os presos não pararam na mesma altura'
    assert altura.mean() < abaixo.min(), 'a junta presa não subiu até a linha'
    recuo = CUT_INSET_SHOULDER_WIDTHS * LARGURA_DE_OMBROS
    print(f'  uma linha separa preso de intocado    {int(movidos.sum())} presos, '
          f'recuo {recuo:.4f}  OK')


def verifica_corte_fixo_com_corpo_em_movimento():
    """Verificação 8: o corpo anda, a linha não."""
    deriva = 0.10
    original = uma_janela_sentada(deriva)['keypoint_labels']
    np.random.seed(3)
    cortada = SimulatedEstimatorNoise(frame_cut_prob=1.0).transform(
        uma_janela_sentada(deriva))['keypoint_labels']

    presos_sempre = np.all(_deslocamento(cortada, original) > 0.0, axis=0)
    assert presos_sempre.any(), 'nenhuma junta ficou presa a janela inteira'

    variacao_do_preso = cortada[:, presos_sempre, 1].std(axis=0).max()
    variacao_do_ombro = original[:, SHOULDER_INDICES, 1].std(axis=0).max()
    assert variacao_do_preso < 4 * CUT_JITTER_SHOULDER_WIDTHS * LARGURA_DE_OMBROS
    assert variacao_do_preso < variacao_do_ombro / 2, (
        'a altura do preso acompanhou o corpo; a linha não está fixa')
    print(f'  linha fixa enquanto o corpo desce     preso {variacao_do_preso:.4f} '
          f'contra ombro {variacao_do_ombro:.4f}  OK')


def verifica_quadril_cortado():
    """Verificação 9: o caso de mesa existe, e é o que o v2 nunca viu.

    Nenhum grupo de `KEYPOINT_GROUPS` contém os índices 11 e 12, de modo que o
    v2 jamais recebeu um quadril escondido. Sob esse corte aplicado ao H3WB com
    verdade de campo ele erra 507mm nos quadris. Aqui só se mede que o corte de
    mesa e o de retrovisor ocorrem os dois, e com frequência comparável.
    """
    transformacao = SimulatedEstimatorNoise(frame_cut_prob=1.0)
    original = uma_janela_sentada()['keypoint_labels']

    quadril_cortado = joelho_cortado = 0
    janelas = 400
    np.random.seed(11)
    for _ in range(janelas):
        cortada = transformacao.transform(
            uma_janela_sentada())['keypoint_labels']
        movidos = _deslocamento(cortada, original)[0] > 0.0
        quadril_cortado += bool(movidos[HIPS].all())
        joelho_cortado += bool(movidos[KNEE_INDICES].all())

    fracao_mesa = quadril_cortado / janelas
    assert joelho_cortado == janelas, 'o joelho deveria cair sempre abaixo do corte'
    assert 0.25 < fracao_mesa < 0.75, (
        f'o corte de mesa ficou desbalanceado: {fracao_mesa:.2f} das janelas')
    print(f'  quadril cortado (caso de mesa)        {fracao_mesa:.2f} das janelas  OK')


def verifica_teto_de_confianca():
    """Verificação 10: o teto do contrato é respeitado, com a minoria prevista."""
    transformacao = SimulatedEstimatorNoise(
        frame_cut_prob=1.0, unobserved_confidence=UNOBSERVED_CONFIDENCE_CAP)
    original = uma_janela_sentada()['keypoint_labels']

    np.random.seed(5)
    cortada = transformacao.transform(uma_janela_sentada())['keypoint_labels']
    movidos = _deslocamento(cortada, original) > 0.0

    confianca_cortada = cortada[..., 2][movidos]
    confianca_mantida = cortada[..., 2][~movidos]

    assert confianca_mantida.min() >= CONFIDENCE_OBSERVED[0]
    assert confianca_cortada.max() <= CONFIDENCE_EXTRAPOLATED[1]

    sob_o_teto = (confianca_cortada <= UNOBSERVED_CONFIDENCE_CAP).mean()
    esperado = 1.0 - CUT_LOOKS_OBSERVED_PROB
    assert abs(sob_o_teto - esperado) < 0.05, (
        f'{sob_o_teto:.2f} das cortadas sob o teto, esperado {esperado:.2f}')
    acima = confianca_cortada[confianca_cortada > UNOBSERVED_CONFIDENCE_CAP]
    assert acima.min() >= CONFIDENCE_EXTRAPOLATED[0], (
        'a minoria que parece observada caiu fora da faixa de extrapolado')
    print(f'  teto de confiança respeitado          {sob_o_teto:.2f} sob '
          f'{UNOBSERVED_CONFIDENCE_CAP}  OK')


def verifica_alvos_intocados():
    """Verificação 11: o corte corrompe a entrada, nunca a supervisão.

    Cortar o alvo junto transformaria o experimento em outro: a rede deixaria de
    ser cobrada justamente pelas juntas que ela precisa aprender a inferir sem
    ver. O que some é a observação, não a resposta.
    """
    supervisao = ('lifting_target_label', 'lifting_target_visible',
                  'lifting_target_weight')
    for transformacao in (
            SimulatedEstimatorNoise(prob=1.0),
            SimulatedEstimatorNoise(
                frame_cut_prob=1.0,
                unobserved_confidence=UNOBSERVED_CONFIDENCE_CAP)):
        np.random.seed(2)
        results = um_resultado_com_alvos()
        esperado = {chave: results[chave].copy() for chave in supervisao}
        saida = transformacao.transform(results)

        for chave, valor in esperado.items():
            assert np.array_equal(saida[chave], valor), f'{chave} foi alterado'
        assert saida['lifting_target_weight'].min() == 1.0, (
            'o peso da perda caiu; as juntas cortadas deixaram de ser cobradas')
    print('  alvo e pesos da perda intocados                      OK')


def verifica_config_casado_com_o_contrato():
    """Verificação 12: o config do v3 repete o teto; os dois não podem divergir.

    O valor aparece em dois lugares --- a constante deste módulo, que o painel
    importa em inferência, e o literal do config, que o MMEngine não consegue
    importar. Divergirem significaria treinar com uma faixa e inferir com outra,
    que é exatamente o modo de falha que o teto existe para fechar, e nenhum
    erro apareceria: o treino rodaria inteiro e só a medição final denunciaria.
    """
    from mmengine.config import Config

    raiz = Path(__file__).resolve().parents[1]
    cfg = Config.fromfile(
        str(raiz / 'configs' / 'lift3d_dstformer_h3wb_robusto_v3.py'))
    simulacao = next(passo for passo in cfg.train_pipeline
                     if passo['type'] == 'SimulatedEstimatorNoise')

    assert simulacao['unobserved_confidence'] == UNOBSERVED_CONFIDENCE_CAP, (
        f"o config treina com {simulacao['unobserved_confidence']} e o painel "
        f'infere com {UNOBSERVED_CONFIDENCE_CAP}')
    assert 0.0 < simulacao['frame_cut_prob'] < 1.0, (
        'o corte precisa ser sorteado: em 0 a rede não o vê, em 1 ela nunca vê '
        'o corpo inteiro')
    print(f'  config e contrato de inferência casam  teto '
          f'{UNOBSERVED_CONFIDENCE_CAP}  OK')


def main():
    verifica_modo_de_grupo()
    verifica_identidade_com_a_versao_do_v2()
    verifica_linha_de_corte()
    verifica_corte_fixo_com_corpo_em_movimento()
    verifica_quadril_cortado()
    verifica_teto_de_confianca()
    verifica_alvos_intocados()
    verifica_config_casado_com_o_contrato()

    print('\na simulação produz o canal de confiança informativo que faltava,')
    print('e o corte mostra à rede o quadril escondido que o v2 nunca viu')


if __name__ == '__main__':
    main()
