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
| 1 — aquisição e pré-processamento | Parcial: captura, cinza e calibração da webcam (0,414 px de reprojeção); falta a câmera infravermelha própria |
| 2 — estimação 2D de 133 keypoints | Parcial: YOLO26n-pose + RTMW-x, estimador escolhido pela montagem; meta de whole-body AP não atingida |
| 3 — lifting 2D→3D temporal | Parcial: DSTFormer de 16 quadros, causal, float16, adaptação veicular; falta validação contra referência independente |
| 4 — painel de validação | Concluído: 2D, 3D de corpo inteiro com observado e previsto, confiança e latência por estágio |

Números medidos, com as condições de medição, estão no `CLAUDE.md` e em
`results/`. Os principais:

- **Whole-body AP 0,6930** em COCO-WholeBody grayscale, com caixas de ground
  truth e flip test, após adaptação de baixo posto sobre o RTMW-x. A meta era
  0,70 e não foi atingida.
- **38,96 mm de MPJPE full-body** no H3WB, sujeito retido S7, a partir de 2D de
  ground truth com janela de 16 frames. Não comparável ao benchmark do H3WB,
  que avalia frame único — ver `CLAUDE.md`.
- **Tempo real cumprido**: 30,4 ms medianos, 32,9 FPS, no caminho completo com o
  lifting em float16 na RTX 5060. O painel ao vivo, que soma desenho e câmera,
  roda a 19,4 FPS.
- **Bateria de validação**: 7 de 8 critérios; o que falha é o whole-body AP.

## Rodar

```bash
source venv/bin/activate

# Painel de validação com a webcam, usando o modelo corrente
python scripts/run_panel.py

# O mesmo painel sobre um vídeo do Drive&Act, na montagem de retrovisor
python scripts/run_panel.py --source <video>.mp4 --montagem retrovisor \
    --calibracao <video>.calibration.json

# Bateria de validação completa (oito critérios do Projeto Físico)
python scripts/run_validation_battery.py

# Avaliar um checkpoint 2D
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
    --ckpt <checkpoint> --data-root data/processed/grayscale/ --tag <nome>

# Testes: só CPU, cada arquivo roda sozinho
for t in tests/test_*.py; do python "$t"; done
```

**Não rode o painel com um treino em andamento.** A disputa pela GPU derruba a
taxa para cerca de um terço da real, e o número exibido induz a erro.

Os `scripts/run_*.sh` são as filas que produziram as medições dos documentos.
Ficam como registro do comando exato de cada número, e esperam a GPU livre antes
de começar.

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
```

`CLAUDE.md` é a documentação corrente: ambiente, armadilhas conhecidas do
MMPose, números medidos e as regras de arquitetura do projeto. Leia-o antes de
mexer no código.
