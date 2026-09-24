# TCC — Estimação de pose 3D full-body para monitoramento veicular (código)

Implementação do TCC de Davi Baechtold Campos (Eng. Computação, PUCPR).
Orientador: Prof. Dr. Alceu de Souza Brito Junior. Coorientador: Prof. Dr. Alessandro Zimmer.

Estimar pose 3D de corpo, face e mãos de ocupantes de veículo, a partir de uma
única câmera infravermelha, em tempo real. O documento que especifica o sistema
vive em outro repositório: `~/Documents/Projeto-Fisico` (Projeto Físico + monografia).
**Mudança de arquitetura, meta ou métrica aqui exige atualizar o documento lá, e vice-versa.**

## Ambiente

```bash
cd ~/Documents/TCC/Project && source venv/bin/activate
```

PyTorch 2.8+cu128 · MMPose 1.3.2 · MMCV 2.1.0 · MMEngine 0.10.7 · MMDet 3.2.0

**Hardware e a assimetria que ele impõe.** Casa: RTX 5060, 8 GB, i5-14400F.
Faculdade, sob demanda junto ao coordenador: 8× RTX 4090 de 24 GB.

*Treinar* pode acontecer no cluster, então modelos pesados estão liberados.
*Inferir* precisa caber em 8 GB e sustentar 20 FPS na 5060 — **a apresentação
final roda obrigatoriamente na 5060**. Esse é o filtro que decide se uma
arquitetura entra no projeto. Preferência explícita: fazer tudo em casa; o
cluster é exceção, não caminho padrão.

`mmcv` está sem extensões compiladas. Isso é inofensivo para o pipeline top-down
(nenhuma op custom é usada), mas gera dezenas de `RuntimeWarning` no stdout —
filtre a saída ao ler logs, não tente "consertar".

Checkpoints do OpenMMLab (2023) contêm objetos numpy e o PyTorch ≥ 2.6 usa
`weights_only=True` por padrão, rejeitando-os. Todo entrypoint que carrega
checkpoint precisa do patch de `torch.load` — veja `scripts/eval_checkpoint.py`.

### Armadilhas do MMPose/MMEngine — conferir antes de culpar o próprio código

Todas já custaram horas aqui, e todas são **silenciosas**: o treino roda, a perda
cai, e só uma métrica denuncia. Lista completa na memória
`mmpose-armadilhas-conhecidas`. As três que mais mordem:

- **`Runner.from_cfg` não carrega `load_from`** — quem carrega é `train()`, que
  antes ainda chama `init_weights()`. Qualquer modificação do modelo feita entre
  os dois é desfeita sem aviso.
- **`Runner.load_checkpoint` liga `_has_loaded`**, e o `load_or_resume` seguinte
  retorna sem fazer nada. Carregar checkpoint antes do `train()` desliga a
  retomada e o treino recomeça da época 1 em silêncio.
- **`requires_grad=False` não congela BatchNorm** — a camada segue normalizando
  pelo lote e sobrescrevendo as estatísticas acumuladas.

Use bfloat16, nunca float16: com fp16 o recorte de gradiente corrompe os pesos
quando a norma transborda. E o shim de `src/models/bf16_compat.py` precisa estar
carregado, porque o MMPose não converte bfloat16 para NumPy.

## Estado atual (set/2026)

| Componente | Estado |
|---|---|
| Dataset COCO-WholeBody grayscale | Pronto: 118.287 treino / 5.000 val |
| Módulo 2 — estimação 2D top-down | Funcional (YOLO26n-pose + RTMW-x). Estimador por montagem: Etapa 2 na mesa, Etapa 3 com ensaio no retrovisor (0,0295 no Drive&Act) |
| Módulo 1 — aquisição | Parcial: captura e calibração OK (fx 959,4, reprojeção 0,414px). **A distância ao ocupante é o parâmetro mais frágil do sistema** — dela dependem a escala de entrada e a de saída do lifting, e os 0,97m informados na demo são desmentidos pela pose reconstruída (interpupilar 43,3mm; a 1,40m daria 63,7mm). Medir com trena. |
| Módulo 3 — lifting 3D | DSTFormer 42,4M params sobre H3WB; batch 4 é o teto dos 8 GB. Checkpoint corrente: `work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth` (treino com corte de quadro) |
| Módulo 4 — visualização | Painel completo: 2D, 3D de corpo inteiro com previsto distinto de observado, escala métrica fixa, **filtro por observação** (junta prevista estabilizada: tremor das pernas 70,97 → 10,61mm sem damping do movimento real), métricas por região coerentes com o desenho |
| Drive&Act | Vídeos e anotações baixados; conversor escrito e validado |
| H3WB (lifting 3D) | Baixado e convertido: 60k treino / 20k teste, 133 keypoints |
| Human3.6M (imagens) | Não necessário — a tarefa 2D→3D usa só coordenadas |

### Números medidos — não sobrescrever com estimativa

Whole-body AP em COCO-WholeBody val, 133 keypoints, **bbox de ground truth**,
flip test ligado. Fonte: `results/baselines/`.

AP em COCO-WholeBody val2017, bbox de GT, flip test ligado:

| Configuração | AP gray | AP RGB | gap |
|---|---|---|---|
| **RTMW-x + LoRA, época 5** (checkpoint atual) | **0,6930** | — | — |
| RTMW-x 384×288 sem treino | 0,6857 | 0,7273 | **−5,7%** |
| RTMPose-m 256×192 sem treino | 0,5255 | 0,6039 | −13,0% |
| RTMPose-m, fine-tuning de 10 epochs a LR 5e-4 | 0,5137 | — | — |
| RTMPose-m, treino de 50 epochs a partir do checkpoint body7 | 0,4373 | — | — |

O checkpoint fundido está em
`work_dirs/rtmw_x_gray_lora/best_coco-wholebody_AP_epoch_5_merged.pth`. É ele que
o `configs/rtmw_x_driveact_ft.py` carrega, e é ele que se usa para inferência —
o checkpoint bruto tem nomes de camada adaptados e não carrega num config comum.

Throughput na 5060, lote 1, fp32, mediana de 100 iterações após 20 de
aquecimento, com `torch.cuda.synchronize()` a cada iteração:

| Estágio | Custo (um ocupante, 23/09/2026) |
|---|---|
| Detector YOLO26n-pose | 4,5–4,8 ms (RTMDet-nano, anterior: 7,3–7,7 ms) |
| Pose RTMW-x 384×288 | ~15 ms **por caixa** |
| Flip test | +12 ms por pessoa |
| Caminho completo com lifting (bateria) | 47,4 ms, 21,1 FPS |

O custo do detector quase não depende do tamanho do quadro (ambos redimensionam
por dentro). `benchmark_throughput.py` agora grava `stages_median_ms`, e é daí
que saem os números por estágio. Flip test é para avaliação: no caminho completo
levaria a ~59 ms, 17 FPS.
**O "24,0 FPS" citado antes não tinha condição declarada e foi descartado.**

Lifting 3D, H3WB, sujeito retido S7, 2D de GT, janela de 16 frames, 30 épocas:
38,96mm full-body, 42,34mm corpo, 9,58mm face, **80,50mm mãos** (dominam o erro).

**O erro das mãos é o erro do punho, e é profundidade.** Decomposto no S7 com
`scripts/measure_hand_error.py`: absoluto 84,5mm, sem o erro do punho 35mm,
**só a forma 11mm**. A cadeia cresce do tronco para a ponta (ombro 28, cotovelo
63, punho 73, mão 84,5) e de 97 a 99% do erro é **profundidade** — no plano da
imagem tudo erra 4 a 6mm, porque a entrada 2D é ground truth. Três alavancas
testadas e descartadas antes de gastar GPU: repesar a perda (as mãos já são 67%
da massa de erro), paralaxe (correlação movimento×erro −0,035), encurtamento
(−0,037). A quarta rende 8%: corrigir o comprimento de osso da cadeia inteira
leva o punho de 72,1 para 66,5mm, e a meta de 60mm exigiria 30%. **Meta
registrada como não atingida, com o motivo medido.**
A curva ainda descia na época 30 — mais épocas é alavanca disponível aqui, ao
contrário do LoRA, que saturou.

Erro por posição na janela (200 janelas): central 39,35mm, causal 42,86mm. O
causal custa 3,51mm e elimina 267ms de atraso — é o adotado.

**A escala métrica exige calibração da câmera.** A saída bruta da cabeça mede
metade do tamanho real; quem restaura é o decodificador, lendo `factor` do
metainfo. Por isso `SequenceLifter` chama `model.predict`, não
`head(backbone(x))`. O fator vem da geometria da câmera: com o verdadeiro o erro
é 43,4mm, com a mediana do H3WB é 72,7mm. A calibração do Módulo 1 é
pré-requisito, não acabamento.

**Não compare os 38,96mm com os 88,3mm do benchmark do H3WB.** O benchmark é
frame único e usa outro conjunto. A comparação válida é o lifting temporal de
17 juntas no Human3.6M (40,9mm MixSTE), e os 42,34mm de corpo caem nessa faixa.

**O proxy grayscale não cobre o domínio real**, medido sobre 300 imagens de
cada: COCO em cinza tem média 105,3 e desvio 56,4; o NIR do Drive&Act tem 29,6
e 31,0. Três vezes e meia mais escuro, metade do contraste. É o que explica a
saturação do LoRA e o que justifica a Etapa 3.

**A calibração corrige a saída do lifting, não a entrada.** O codec normaliza o
2D pela largura do quadro, de modo que a rede recebe `2·fx/(Z·W)` unidades
normalizadas por metro: 0,447 no H3WB, 1,545 numa webcam a 0,97m — 3,46× fora da
distribuição de treino, e o DSTFormer não normaliza escala internamente. Medido
no S7 com a geometria da webcam: **276,3mm de erro contra 48,0mm** depois de
remapear a entrada para a geometria de treino (referência 47,5mm). Ao vivo o
tremor do tronco cai 2 a 3 vezes. `CameraView` em `src/models/sequence_lifter.py`
faz o remapeamento; sem calibração o caminho antigo continua, declarado como
aproximado.

**O lifting corrente é o treinado com corte de quadro (v3).** Protocolo de corte
sobre o S7 (`scripts/measure_lifting_truncation.py`), condição `mesa`, que é o
enquadramento da webcam:

| Modelo | MPJPE | Visíveis | Quadris | Pernas | Tronco (GT 453,8mm) |
|---|---|---|---|---|---|
| Base | 140,7mm | 39,7mm | 243,9mm | 747,2mm | 261,0mm |
| Simulação (v2) | 103,7mm | 40,2mm | 495,8mm | 237,2mm | 181,6mm |
| **Corte (v3)** | **55,7mm** | **29,5mm** | **72,0mm** | **117,5mm** | **464,8mm** |

Na entrada íntegra não custa nada (39,6 contra 39,9mm; 36,32 na validação
oficial). No Drive&Act os dois empatam dentro da incerteza da referência
(81,79 contra 79,09mm de PA-MPJPE) e a coerência de osso melhora de 29,78 para
25,20mm com movimento igual.

**O teto de confiança é contrato entre treino e inferência.** `UNOBSERVED_CONFIDENCE_CAP = 0,3`
em `src/data/estimator_noise.py`: o painel rebaixa a confiança das juntas que
sabe não ter observado, e o v3 treinou vendo essa faixa. Aplicá-lo a um
checkpoint que não treinou com ele **piora** — v2 vai de 495,8 para 609,9mm no
quadril. Por isso ele anda junto do checkpoint (`--teto-confianca`), não do painel.

**A simulação descreve parte do mecanismo, e a transferência é parcial.** Na
gravação real o v3 corta pela metade a incoerência de forma (0,3625 → 0,1800) e
traz a largura de quadril para a faixa adulta, mas o tronco previsto fica curto —
contra 464,8mm no corte simulado. O estimador real prende o quadril na borda (o
que a simulação reproduz) e espalha joelhos e tornozelos sobre tronco e braços
(o que ela não reproduz). Medido em `results/posicao_ausentes_*.json`: o quadril
fica junto da borda em 98% dos quadros na mesa, o joelho em 43 a 53%, o
tornozelo em 31 a 49% e os pés em 6 a 32% — o resto cai **sobre o corpo
visível**. No retrovisor a corrupção é outra: as pernas descem numa cadeia
plausível e quase não tocam o corpo. São dois mecanismos, e a simulação descreve
um.

**A escala métrica está verificada contra referência física: erro de 1,7%.**
Gravação com o tabuleiro de calibração ao lado do rosto: solvePnP com os
intrínsecos completos dá 1110mm de distância; os 55,94px entre os centros dos
olhos dão 64,7mm de interpupilar real por pinhole; o lifting reconstrói 63,6mm.
**Dado o Z correto, a escala do sistema está certa** — e o Z é a única grandeza
que a câmera monocular não observa, então é ele que precisa ser medido em cada
montagem. A gravação de 12/09 reconstruía 50,0mm de interpupilar porque a
distância declarada (0,97m) não era a real, que os 64,7mm implicam ter sido
~1,51m. Esconder o quadril **não** altera a face (62,3 contra 64,3mm na mesma
gravação): o corte quebra o corpo, não a escala.

**O que o corte quebra é a perna.** Com as pernas observadas o lifting
reconstrói coxa 411,0 e canela 455,3mm, dentro da faixa adulta; cortadas, viram
574,6 e 324,6mm. A razão coxa/tronco degrada com o que se esconde: 1,24 com
quadril visível, 1,39 sem ele, 1,57 no corte alto — contra 0,85 da anatomia.

**O Módulo 3 recebeu a adaptação de domínio que nunca teve.**
`configs/lift3d_veicular.py`: metade do lote é Drive&Act com 2D do estimador
real e alvo da referência 3D (92.080 quadros, 11.481 janelas, extraídos por
`scripts/build_driveact_lift_dataset.py`), metade H3WB por ensaio — sem ele a
supervisão parcial (23 de 133 keypoints) repetiria o esquecimento da Etapa 3. A
corrupção simulada é desligada sobre o Drive&Act, que já chega corrompido de
verdade (marca `corrupcao_real`).

| | v3 (corte) | veicular |
|---|---|---|
| Drive&Act PA-MPJPE | 81,79mm | **41,49mm** |
| Drive&Act MPJPE | 183,71mm | **60,69mm** |
| Coerência de osso | 25,20 | **19,58** (com mais movimento: 9,18 contra 7,99) |
| H3WB whole | **36,32mm** | 39,77mm |
| Webcam: canela / interpupilar | **297,5 / 62,1mm** | 18,9 / 35,8mm |

**Duas ressalvas.** O modelo é treinado contra a mesma referência que o avalia,
então os 41,49 medem concordância com a triangulação do OpenPose, não acurácia;
a coerência de osso, que não usa referência, é a evidência limpa. E **o treino
tinha um defeito de perda** (achado em 23/09): a `MPJPEVelocityJointLoss` só aplica
o `lifting_target_weight` com `use_target_weight=True`, e o padrão é `False`. Os
121 pontos sem referência de cada janela do Drive&Act têm alvo idêntico (dispersão
0,00mm) e foram supervisionados contra ele — a rede colapsa face, mãos e pernas.
A canela de 18,9mm na webcam, lida antes como "quebra fora da montagem", é esse
defeito — e no próprio Drive&Act o modelo reconstrói interpupilar de **3,3mm**
(`results/lifting_driveact_veicular_defeito.json`, com a geometria dos 133
pontos que o validador passou a medir). **Ligar `use_target_weight` não basta**:
nesse caminho a perda do MMPose multiplica as 15 velocidades pelos 16 pesos e
quebra. A perda agora é `WeightedMPJPEVelocityLoss` (`src/models/lifting_loss.py`),
com `tests/test_lifting_loss.py` (coincide com a original a peso 1; peso zero não
muda perda nem gradiente) e `tests/test_partial_supervision_loss.py` (exige essa
perda em todo treino com alvo parcial). Retreino em
`work_dirs/lift3d_veicular_peso` por `scripts/run_lifting_veicular_peso.sh`; até
ele fechar, o checkpoint veicular vale só para os 12 pontos corporais. O
lifting continua escolhido pela montagem (`LIFT_CHECKPOINT_BY_MOUNTING`), decisão a
reavaliar com o modelo retreinado.

**Armadilha de leitura que me custou quatro medições:** a linha de validação do
MMEngine traz `MPJPE` e `P-MPJPE` juntas. Uma regex gulosa captura a segunda, e
eu "descobri" mãos a 16,8mm — que são o P-MPJPE, isto é, a forma da mão já
medida em 11mm. Ler o campo, não a posição.

**Simular a colocação medida não transferiu — resultado negativo.** O v4
(`cut_placement='medido'`) reproduz a distribuição medida de onde cada junta
cortada cai, e não melhora: no Drive&Act 81,08 contra 81,79mm de PA-MPJPE
(dentro da incerteza de 30,1mm da referência) e na gravação com distância
medida as proporções da perna **pioram** — coxa 593,7 contra 527,9, canela
286,7 contra 340,6. **O painel segue com o v3.** O que a medição cruzada achou,
e que vale como método: sob o mecanismo alheio o v4 degrada 4% (56,5 contra
59,0) e o v3 degrada 33% (74,1 contra 55,7) — a mistura de três modos ensina
tolerância a corrupções que ela não contém. Robustez, não acurácia.

**A Etapa 3 esqueceu face e mãos, e o checkpoint de operação é o da Etapa 2.**
Medido no COCO em cinza, onde face e mãos têm anotação, caixa de GT, 300
instâncias: whole-body AP de **0,6931 para 0,2330**; erro da face (normalizado
pela interocular) de 0,0359 para 0,2198, seis vezes; mãos de ~0,085 para ~0,205,
duas vezes e meia; corpo de 0,0408 para 0,0805. A resposta média na face cai de
9,68 para 6,82. É o mesmo peso zero que produziu a impunidade nas pernas, com
efeito oposto: onde a junta **não** está na imagem o modelo fica confiante e
errado; onde ela **está**, ele esquece. Instrumento: `scripts/measure_region_error.py`.

**O ensaio corrigiu.** `configs/rtmw_x_driveact_ensaio.py` intercala o Drive&Act
com o subconjunto do COCO que anota face ou mãos (77.214 das 262.465 instâncias;
no conjunto integral só 29,4% anotam, e o resto dilui o sinal). Parte do
checkpoint da Etapa 2, não do v2. Medido: face de volta a 0,0367 contra 0,0359
da Etapa 2, mãos 0,080/0,090, whole-body AP 0,6848, **ao custo de 0,42px de erro
corporal** (9,94 contra 9,52 da v2, ainda 34% melhor que os 15,04 de partida). A
supervisão de incerteza sobrevive: 7,5% das juntas ausentes acima do limiar
contra 7,4% da v2. Checkpoint em
`work_dirs/rtmw_x_driveact_ensaio/best_torso_px_mean_epoch_2_merged.pth`.

**Confirmado na webcam**, que é onde a v2 falhava: raio dos landmarks de face
36,3px contra 35,2 da Etapa 2 — e 164,5px na v2, com distância interpupilar
reconstruída de 225mm. **O painel passa a escolher o estimador pela montagem**
(`POSE_CHECKPOINT_BY_MOUNTING`): ensaio no retrovisor (9,94 contra 15,04px de
erro corporal), Etapa 2 na mesa (treme menos no domínio em que foi treinado —
face 1,39 contra 3,22mm — embora o ensaio acerte melhor a anatomia). O
checkpoint sem ensaio não entra em montagem alguma.

Quatro conclusões que orientam todo trabalho futuro:

1. **O RTMW-x é o modelo do projeto.** 0,6857 em grayscale sem treino nenhum,
   e cumpre o requisito de 20 FPS na 5060 com folga de três vezes. A adaptação
   por LoRA levou a 0,6930 e saturou; a meta de 70% **não foi atingida**.
2. **O domain gap encolhe com a capacidade do modelo**: 13,0% no RTMPose-m
   contra 5,7% no RTMW-x. É um achado próprio e publicável.
3. **O gap concentra-se em AP.75, não em AP.50.** Perder a cor atrapalha
   *localizar* o keypoint com precisão, não *encontrá-lo*.
4. **Fine-tuning completo a LR alto piora o modelo.** Medido: 0,5255 → 0,5137
   em 10 epochs a 5e-4, por catastrophic forgetting. A adaptação de domínio
   tem que ser por LoRA ou LR muito baixo, nunca por fine-tuning agressivo.

**O detector de operação é o YOLO26n-pose, e a QP1 foi refeita (23/09/2026).**
A QP1 original usava os primeiros 1.500 quadros do val do Drive&Act — **todos de
vp14_run1**, porque o JSON guarda as imagens em ordem de sequência — e o
estimador da mesa. Refeita com `annotated_pairs` (espaçada, `src/data/driveact.py`)
e o estimador do retrovisor, sobre 1.480 quadros das quatro sequências:

| Configuração | Erro | Latência | Caixas espúrias |
|---|---|---|---|
| Caixa de GT | 0,0301 | 15,16 ms | 0 |
| **YOLO26n-pose** | **0,0310** | 20,82 ms | 39 (3%) |
| RTMDet-nano | 0,0395 | 32,41 ms | **846 (57%)** |
| Sem detector | 0,0408 | 15,16 ms | — |

**O RTMDet devolve caixa espúria em 57% dos quadros NIR** (um ocupante só). Cada
uma custa uma passada de pose (era isso os "12 ms", não o detector) e a primeira
caixa nem sempre é o ocupante. Atribuído por medição: mesmo estimador, RTMDet
0,0276 nos primeiros 600 quadros contra 0,0476 espaçados — a amostra escondia
a falha. Sem detector custa 32%, não 82% (o 82% era do estimador de cinza).

A troca **não prejudica face nem mãos** (pergunta de Davi, medida):
`measure_region_error.py --detector` casa a caixa com a anotação por IoU. No
COCO em cinza as medianas empatam e a média da face cai de 0,1186 (RTMDet) para
0,0569 (YOLO) — o RTMDet tem cauda de caixas ruins.

Consequências no código: `src/models/detector_config.py` é o catálogo (sem
dependências: importar `pose_pipeline` para ler três strings custava 3,2 s);
`build_person_detector` em `pose_pipeline`; `instance_error` e `occupant_index`
em `normalized_keypoint_error`. **`build_driveact_lift_dataset.py` prende o
RTMDet** (o modelo veicular foi treinado sobre o 2D dele) e agora escolhe a caixa
do ocupante: o conjunto atual, extraído pela primeira caixa, tem ~1% dos quadros
pareados a caixa espúria (erro > 0,2 tronco), medido sem GPU contra a anotação 2D.

**Estratificação (Fase 2, `measure_stratified_error.py`):** iluminação não é
fator (terço escuro 0,0301 contra claro 0,0318); oclusão é (+32% com 8–10 juntas
visíveis contra 12–14). O Drive&Act não tem luz solar nem túnel.

**Bateria de validação: 7 de 8 passam** (`results/bateria_validacao.json`,
23/09/2026, com YOLO26n-pose). A falha é o critério de precisão full-body: AP
0,6931 contra 0,70 e AR 0,7479 contra 0,75. Tempo real com folga de 5% (21,1
FPS, 47,4 ms medianos, caminho completo em 1280x720; com o RTMDet eram 20,2);
degradação de 8,2% sob oclusão do punho (teto 15%); **a janela temporal reduz o
tremor em 29,1%** contra a mesma rede sem contexto (21,7% com o RTMDet); mil
quadros sem exceção e sem crescimento de memória. O critério 1b cita agora o
checkpoint de operação (ensaio, 0,0295) — antes citava a Etapa 3 v2 (0,0282),
aposentada por esquecer face e mãos.

A primeira execução reprovou dois testes por defeito do arnês, não do sistema:
o teste veicular usava o checkpoint da montagem de mesa (68% de degradação, que
media troca de modelo) e o de coordenadas contava as pernas extrapoladas como
keypoints fora do quadro. **Testar o instrumento antes de acreditar nele.**

Sempre declare se um AP usa bbox de ground truth ou de detector. A diferença é
de ~2 pontos e comparar as duas condições silenciosamente invalida o resultado.

## Manter o documento vivo — regra obrigatória

Este repositório implementa o que o Projeto Físico especifica, e os dois andam
juntos. **Toda medição, decisão de arquitetura ou descoberta que contradiga o
documento obriga a atualizá-lo na mesma sessão**, em `~/Documents/Projeto-Fisico`.

Vale especialmente para resultado negativo: um treino que piorou o modelo ou uma
meta que se revelou inviável são conteúdo do TCC, não fracasso a esconder.

Números medidos aqui e citados lá precisam bater. Quando divergirem, o valor
medido ganha, e o documento é corrigido — nunca o contrário.

## Comandos

```bash
# Medir um checkpoint em qualquer domínio (é assim que se isola o domain gap)
python scripts/eval_checkpoint.py \
  --ckpt checkpoints/rtmpose-m_wholebody_official_256x192.pth \
  --data-root data/processed/grayscale/ --tag official_gray

# Treino de adaptação de domínio. --epochs reescala o cronograma
# (cosine, troca de pipeline) em vez de truncá-lo.
python scripts/train_wholebody.py \
  --config configs/rtmpose_m_wholebody_gray_ft.py [--epochs 10]

# Painel de validação ao vivo, com os modelos correntes por padrão
python scripts/run_panel.py [--montagem mesa|retrovisor] [--distancia 1.35]

# Protocolo de corte: quanto o lifting erra nas juntas que a câmera não vê.
# A condição `mesa` é o mecanismo que treina o v3 — para ele mede aderência ao
# próprio treino, não generalização. A evidência independente é o Drive&Act.
python scripts/measure_lifting_truncation.py --lift-cfg <cfg> --lift-ckpt <ckpt> --tag <nome>

# Qualidade ao vivo sobre um vídeo, sem referência: tremor por região, controle
# de movimento, coerência de osso e plausibilidade anatômica. O passo 2D fica em
# cache, então comparar checkpoints roda o estimador uma vez só.
python scripts/measure_live_quality.py --tag <nome> [--lift-ckpt <ckpt>] [--sem-filtro]

# Calibração da webcam (Logitech C922, 1280x720). Já feita em 12/09/2026:
# fx 959,4  fy 957,9  centro (618,1; 342,7)  reprojeção 0,414px em 21 vistas.
# Refazer só se trocar a câmera, a resolução ou o foco.
python scripts/calibrate_camera.py --lado-quadrado 0.026

# Erro por região anatômica com ground truth (face pela interocular, mãos pela
# diagonal, corpo pelo tronco). O whole-body AP é um número só e não diz o que
# quebrou — foi assim que o esquecimento da Etapa 3 passou despercebido.
python scripts/measure_region_error.py --tag <nome> --ckpt <checkpoint>

# Bateria de validação inteira, com critério de aceite por teste. Leva ~6min.
# Três testes são citados de medições já gravadas; os outros são medidos na hora.
python scripts/run_validation_battery.py

# Confere se o documento cita as medições correntes (a regra dos dois repos)
python scripts/check_document_numbers.py

# Taxa de processamento, com as condições registradas junto do resultado
python scripts/benchmark_throughput.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
  --ckpt <checkpoint> --tag <nome> --image <imagem com pessoas> [--flip-test] \
  [--detector yolo26n-pose|rtmdet-nano|nenhum]
```

**Não rode o painel com um treino em andamento**: a disputa pela GPU derruba a
taxa para cerca de um terço da real (63ms por quadro contra 20ms com a GPU
livre), e o número exibido induz a erro numa demonstração.

```bash
# Fila de treinos longos, resiliente a travamentos da máquina.
# Relançar após uma queda continua de onde parou; o progresso fica em arquivo.
setsid nohup ./scripts/run_overnight.sh > work_dirs/logs/fila.log 2>&1 &
```

**Lance todo treino longo por aí.** A máquina trava sozinha, por causa alheia ao
projeto — 26% dos boots terminam em congelamento. Diagnóstico completo na
memória `hardware-instabilidade-do-pc`. Logs em `work_dirs/logs/`, que sobrevive
à troca de sessão, ao contrário do diretório temporário.

### Recuperação automática — três quedas, três respostas

| Queda | Quem recupera |
|---|---|
| Kernel trava por completo | Watchdog da placa (`iTCO_wdt` mais `RuntimeWatchdogSec=60s` no systemd), que reinicia por hardware em 60s |
| GPU sai do barramento com o sistema de pé (21/09/2026) | `scripts/vigia_hardware.sh` detecta e tenta reiniciar; sem uma regra de reinício sem senha em `/etc/sudoers.d`, ela apenas registra no log |
| A fila morre e a GPU continua sadia | A mesma vigia relança; o treino retoma da última época pelo `--resume` |

Depois de qualquer reinício, o `@reboot` do crontab roda
`scripts/retomar_apos_reboot.sh`, que lê o marcador `work_dirs/logs/fila_pendente`,
relança a fila e sobe a vigia junto.

A vigia tem dois modos: sem argumento faz uma verificação (forma de cron), com
um intervalo em segundos fica em laço até a fila concluir. Ela **não** age sem
marcador de fila — máquina ociosa não é reiniciada — e tem teto de três
reinícios, para que defeito permanente não vire laço de boot.

`data/`, `checkpoints/`, `work_dirs/` e `venv/` são ignorados pelo git.
`results/` **não é** — os JSON de métrica ali sustentam afirmações do documento.

---

# Arquitetura e Clean Code — regra obrigatória, sem exceção

Vale para todo desenvolvimento daqui em diante, em código novo e em qualquer
arquivo que se encoste. Referência: SOLID + Clean Code
(https://gist.github.com/danielschmitz/95c6eb40a3845f89498a3c748e932f44).

## MVC — onde cada coisa mora

MVC nasceu para aplicações web; aqui a separação é a mesma, com os nomes
traduzidos para o domínio de visão computacional.

| Camada | Responsabilidade | Onde fica |
|---|---|---|
| **Model** | Dados e regra de negócio: arquiteturas, datasets, transforms, funções de perda, métricas, laços de treino e avaliação. Não sabe que existe terminal, janela ou argumento de linha de comando. | `src/models/`, `src/data/`, `src/training/`, `src/evaluation/` |
| **View** | Apresentação. Recebe resultado pronto e só desenha: overlay de keypoints, esqueleto 3D, gráficos, tabelas de métrica, vídeo de saída. Não calcula nada que valha a pena testar. | `src/visualization/` |
| **Controller** | Recebe entrada (argumentos de CLI, config), chama o Model, entrega o resultado à View. **Fino por definição.** | `scripts/` |

`configs/` é configuração declarativa do MMEngine, não código: sem lógica, sem
`if`, sem cálculo derivado que dependa de estado de execução.

Se um script em `scripts/` tem regra de negócio, ela está no lugar errado. Se
uma função de desenho decide *o que* é um keypoint válido em vez de só desenhar
o que recebeu, ela pertence a `src/evaluation/`.

## Os princípios, e como aplicá-los aqui

- **Responsabilidade única** — o sinal é quantas responsabilidades, não quantas
  linhas. Uma função que desenha um esqueleto de 133 keypoints pode passar de 50
  linhas sem ser ruim; uma de 30 que faz carregamento + inferência + desenho é ruim.
- **DRY** — o mesmo bloco copiado em dois lugares vira função. Dois scripts que
  montam o mesmo pipeline de inferência compartilham o módulo que o monta.
- **KISS** — a solução simples que funciona ganha da elegante que ninguém entende
  às 3 da manhã.
- **Nomes que explicam** — variável e função dizem o propósito. Sem `x`, `tmp`,
  `data2`. Número mágico vira constante nomeada: `bbox_thr=0.5` espalhado pelo
  código vira `DEFAULT_PERSON_SCORE_THRESHOLD`.
- **Poucos parâmetros** — se andam sempre juntos, agrupe num objeto. Config de
  modelo, checkpoint e device são um trio: vira uma dataclass, não três argumentos.
- **Comentário diz POR QUÊ, não O QUÊ** — o código já diz o quê. Comentário bom
  registra a decisão, a armadilha, o bug que motivou aquela linha. Exemplo real
  deste repo: *"inference_topdown já aplica o recorte e a transformação afim;
  recortar antes aplicaria a transformação duas vezes."*
- **Composição antes de herança**; sem função global; sem efeito colateral escondido.

## O que NÃO fazer em nome de arquitetura

Abstração para um caso só é complexidade disfarçada. **Nada de** interface com
uma implementação, factory de um produto, `repositories/` com classes só para
reagrupar funções que já estão agrupadas, ou diretório de schemas para classes
com um consumidor cada. Mover arquivo de lugar sem mover responsabilidade junto
não é refatoração — é rearrumar a mesma bagunça.

## Dívida conhecida — corrigir ao encostar, não em mutirão

Estas violações existem hoje. Não são para consertar de uma vez; são para
consertar quando a tarefa em curso passar por elas.

- **`src/evaluation/evaluate_pose.py` e `evaluate_pose_video.py`** ainda misturam
  as três camadas num arquivo, e comparam RGB contra IR num protocolo que a
  medição de domain gap substituiu. São as próximas a passar pela regra.
- **`scripts/evaluate_accuracy_comparison.py`** funciona, mas chama de
  "bottom-up" o que na verdade é usar o frame inteiro como caixa única. O que
  ele mede — acurácia com caixa do detector contra caixa única — é justamente a
  metade ainda aberta da QP1, então vale reaproveitar em vez de descartar. O
  caminho mais curto para essa resposta, porém, é gerar um arquivo de detecções
  e rodar o `eval_checkpoint.py` duas vezes.
- O remendo do `torch.load` ainda aparece copiado em alguns scripts antigos;
  `src/models/torch_compat.py` é o lugar dele.

## Convenções

- Código, identificadores e mensagens de log em inglês; comentários, docstrings
  e mensagens de commit em português.
- Mensagem de commit explica **por que** a mudança foi feita e o que ela corrige,
  não apenas o que mudou — o diff já mostra o quê.
- Toda métrica reportada vem acompanhada da condição em que foi medida
  (dataset, origem da bbox, flip test).
