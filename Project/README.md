# Estimação de pose 3D full-body para monitoramento de ocupantes veiculares

Sistema de estimação de pose tridimensional de corpo, face e mãos (133
keypoints) a partir de câmera infravermelha monocular, em tempo real.

TCC de Davi Baechtold Campos, Engenharia de Computação, PUCPR.
Orientador: Prof. Dr. Alceu de Souza Brito Junior.
Coorientador: Prof. Dr. Alessandro Zimmer.

A especificação do sistema vive em outro repositório, `~/Documents/Projeto-Fisico`.
**Mudança de arquitetura, meta ou métrica aqui exige atualizar lá.**

## Estado

| Módulo | Estado |
|---|---|
| 1 — aquisição e pré-processamento | Parcial; calibração pendente |
| 2 — estimação 2D de 133 keypoints | Adaptação ao grayscale concluída; adaptação ao Drive&Act em andamento |
| 3 — lifting 2D→3D temporal | Treinado; falta ligar ao painel ao vivo |
| 4 — painel de validação | Funcional em 2D; visualização 3D pendente do Módulo 3 |

Números medidos, com as condições de medição, estão no `CLAUDE.md` e em
`results/`. Os principais:

- **Whole-body AP 0,6930** em COCO-WholeBody grayscale, com caixas de ground
  truth e flip test, após adaptação de baixo posto sobre o RTMW-x.
- **38,96 mm de MPJPE full-body** no H3WB, sujeito retido S7, a partir de 2D de
  ground truth com janela de 16 frames. Não comparável ao benchmark do H3WB,
  que avalia frame único — ver `CLAUDE.md`.
- **Tempo real cumprido**: detector 7,35 ms fixos e pose 12,44 ms por pessoa na
  RTX 5060.

## Rodar

```bash
source venv/bin/activate

# Painel de validação com a webcam, usando o modelo corrente
python scripts/run_panel.py

# Avaliar um checkpoint
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
    --ckpt <checkpoint> --data-root data/processed/grayscale/ --tag <nome>

# Medir a taxa de processamento
python scripts/benchmark_throughput.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
    --ckpt <checkpoint> --tag <nome> --image <imagem com pessoas>
```

**Não rode o painel com um treino em andamento.** A disputa pela GPU derruba a
taxa para cerca de um terço da real, e o número exibido induz a erro.

## Estrutura

```
configs/      Configs do MMPose: avaliação, adaptação de domínio, lifting
scripts/      Controllers — leem argumentos, chamam o modelo, devolvem a view
src/models/   Arquiteturas, LoRA, pipeline de inferência, correções do arcabouço
src/data/     Datasets e conversores (Drive&Act, H3WB)
src/evaluation/  Métricas
src/visualization/  Painel e desenho de esqueleto
tests/        Verificações do que falha em silêncio
results/      Medições versionadas, com o config que produziu cada uma
docs/historico/  Planejamento anterior à pausa; não descreve o sistema atual
```

`CLAUDE.md` é a documentação corrente: ambiente, armadilhas conhecidas do
MMPose, números medidos e as regras de arquitetura do projeto. Leia-o antes de
mexer no código.
