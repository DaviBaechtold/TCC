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
| Módulo 2 — estimação 2D top-down | Funcional (RTMDet-nano + RTMW-x); adaptação de domínio por LoRA em curso |
| Módulo 1 — aquisição | Parcial: captura OK, calibração/undistort pendentes |
| Módulo 3 — lifting 3D | DSTFormer 42,4M params sobre H3WB; batch 4 é o teto dos 8 GB (3,66 GB, 5,1 min/época) |
| Módulo 4 — visualização | Painel de validação funcional (2D, métricas, FPS); visualização 3D pendente do Módulo 3 |
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

| Estágio | Custo |
|---|---|
| Detector RTMDet-nano | 7,35 ms, fixo |
| Pose RTMW-x 384×288 | 12,44 ms por pessoa |
| Flip test | dobra o custo da pose |

Cumpre 20 FPS em três das quatro configurações; falha só com detector + flip
test + duas pessoas (19,1 FPS). Flip test é para avaliação, não para operação.
**O "24,0 FPS" citado antes não tinha condição declarada e foi descartado.**

Lifting 3D, H3WB, sujeito retido S7, 2D de GT, janela de 16 frames, 30 épocas:
38,96mm full-body, 42,34mm corpo, 9,58mm face, **80,50mm mãos** (dominam o erro).
A curva ainda descia na época 30 — mais épocas é alavanca disponível aqui, ao
contrário do LoRA, que saturou.

**Não compare os 38,96mm com os 88,3mm do benchmark do H3WB.** O benchmark é
frame único e usa outro conjunto. A comparação válida é o lifting temporal de
17 juntas no Human3.6M (40,9mm MixSTE), e os 42,34mm de corpo caem nessa faixa.

**O proxy grayscale não cobre o domínio real**, medido sobre 300 imagens de
cada: COCO em cinza tem média 105,3 e desvio 56,4; o NIR do Drive&Act tem 29,6
e 31,0. Três vezes e meia mais escuro, metade do contraste. É o que explica a
saturação do LoRA e o que justifica a Etapa 3.

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

# Inferência em tempo real, multi-pessoa
python src/evaluation/run_realtime.py \
  --cfg <config> --ckpt <checkpoint> \
  --det-cfg configs/detectors/rtmdet_nano_person_infer.py \
  --det-ckpt checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
  --device cuda:0 --source 0
```

```bash
# Fila de treinos longos, resiliente a travamentos da máquina.
# Relançar após uma queda continua de onde parou; o progresso fica em arquivo.
setsid nohup ./scripts/run_overnight.sh > work_dirs/logs/fila.log 2>&1 &
```

**Lance todo treino longo por aí.** A máquina trava sozinha, por causa alheia ao
projeto — 26% dos boots terminam em congelamento. Diagnóstico completo na
memória `hardware-instabilidade-do-pc`. Logs em `work_dirs/logs/`, que sobrevive
à troca de sessão, ao contrário do diretório temporário.

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

- **`src/evaluation/run_realtime*.py` são quatro variantes quase idênticas**
  (`_bottomup`, `_optimized`, `_turbo`, e a base). Violação de DRY. O destino é
  um módulo de pipeline em `src/models/` com as variações como parâmetro, e um
  único controller fino em `scripts/`.
- **`run_realtime.py` mistura as três camadas** num arquivo: `argparse`
  (controller), `inference_topdown` (model) e `draw_keypoints` (view).
- **`scripts/test_bottomup_WORKING.py` e `test_bottomup_debug.py`** — nomes que
  não explicam nada e sugerem código de rascunho versionado.
- **`src/models/` não existe**, embora seja onde o Módulo 3 tem que nascer.

## Convenções

- Código, identificadores e mensagens de log em inglês; comentários, docstrings
  e mensagens de commit em português.
- Mensagem de commit explica **por que** a mudança foi feita e o que ela corrige,
  não apenas o que mudou — o diff já mostra o quê.
- Toda métrica reportada vem acompanhada da condição em que foi medida
  (dataset, origem da bbox, flip test).
