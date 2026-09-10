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
GPU: RTX 5060, 8 GB. CPU: i5-14400F. Treino cabe em ~4 GB com batch 64 em 256×192.

`mmcv` está sem extensões compiladas. Isso é inofensivo para o pipeline top-down
(nenhuma op custom é usada), mas gera dezenas de `RuntimeWarning` no stdout —
filtre a saída ao ler logs, não tente "consertar".

Checkpoints do OpenMMLab (2023) contêm objetos numpy e o PyTorch ≥ 2.6 usa
`weights_only=True` por padrão, rejeitando-os. Todo entrypoint que carrega
checkpoint precisa do patch de `torch.load` — veja `scripts/eval_checkpoint.py`.

## Estado atual (set/2026)

| Componente | Estado |
|---|---|
| Dataset COCO-WholeBody grayscale | Pronto: 118.287 treino / 5.000 val |
| Módulo 2 — estimação 2D top-down | Funcional (RTMDet-nano + RTMPose-m); adaptação de domínio em curso |
| Módulo 1 — aquisição | Parcial: captura OK, calibração/undistort pendentes |
| Módulo 3 — lifting 3D | **Não existe.** Maior risco do projeto |
| Módulo 4 — visualização | Parcial: overlay 2D em OpenCV; falta 3D e painel de métricas |
| Drive&Act | `inner_mirror.zip` (2,2 GB) baixado em ~/Downloads; falta converter |
| Human3.6M | Acesso obtido, download pendente |

### Números medidos — não sobrescrever com estimativa

Whole-body AP em COCO-WholeBody val, 133 keypoints, **bbox de ground truth**,
flip test ligado. Fonte: `results/baselines/`.

| Configuração | Whole AP | Whole AR |
|---|---|---|
| RTMPose-m oficial, RGB | 0,6039 | 0,6670 |
| RTMPose-m oficial, grayscale (zero-shot) | **0,5255** | 0,5964 |
| Treino de 50 epochs a partir do checkpoint body7 | 0,4373 | 0,5287 |

Duas conclusões que orientam todo trabalho futuro:

1. **0,5255 é o piso.** É o que se obtém sem treinar nada. Qualquer treino que
   entregue menos que isso está errado, não "quase lá".
2. O domain gap RGB→grayscale é de **13,0% relativos**, e concentra-se em
   AP.75 (−16,3%) e não em AP.50 (−5,2%): perder a cor atrapalha *localizar* o
   keypoint com precisão, não *encontrá-lo*.

Sempre declare se um AP usa bbox de ground truth ou de detector. A diferença é
de ~2 pontos e comparar as duas condições silenciosamente invalida o resultado.

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
