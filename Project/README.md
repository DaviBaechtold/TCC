# Estimação de pose 3D full-body para monitoramento de ocupantes veiculares

Sistema de estimação de pose tridimensional de corpo, face e mãos (133
keypoints) a partir de câmera infravermelha monocular, em tempo real.

TCC de Davi Baechtold Campos, Engenharia de Computação, PUCPR.
Orientador: Prof. Dr. Alceu de Souza Brito Junior.
Coorientador: Prof. Dr. Alessandro Zimmer.

A especificação do sistema (Projeto Físico) e a monografia vivem em outro
repositório. **Mudança de arquitetura, meta ou métrica aqui exige atualizar lá.**

---

## Para testar: do zero ao painel funcionando

O caminho abaixo leva de uma máquina sem nada do projeto ao painel de validação
rodando com a webcam. Ele foi executado do começo ao fim numa instalação limpa
em 24/09/2026. São cerca de 15 minutos, quase todos de download.

### 1. O que a máquina precisa ter

| Item | Requisito | Por quê |
|---|---|---|
| Sistema | Linux (testado no Linux Mint 22.3, base Ubuntu 24.04) | O painel configura a webcam pelo `v4l2-ctl`, que é do Linux |
| GPU | NVIDIA com pelo menos 8 GB (testado na RTX 5060) | Sem GPU o sistema roda, mas longe de tempo real |
| Driver NVIDIA | 570 ou mais recente | Exigência do CUDA 12.8, o único que suporta a série RTX 50 |
| Python | 3.12 | As versões fixadas em `requirements.txt` foram verificadas nele |
| Disco | ~15 GB livres | Ambiente virtual (~11 GB, quase todo PyTorch) e pesos (~1,1 GB) |
| Webcam | Qualquer webcam USB | Para o painel ao vivo; um vídeo também serve |

Conferir o driver e instalar o que falta do sistema:

```bash
nvidia-smi                      # a linha "Driver Version" precisa ser >= 570
sudo apt install git python3.12-venv v4l-utils ffmpeg
```

O CUDA Toolkit **não** é necessário: o PyTorch traz o próprio runtime CUDA.

### 2. Baixar o código

```bash
git clone https://github.com/DaviBaechtold/TCC.git
cd TCC/Project
```

Todos os comandos a partir daqui rodam dentro de `TCC/Project`.

### 3. Criar o ambiente

```bash
./scripts/instalar_ambiente.sh
source venv/bin/activate
```

O script cria `venv/` e instala tudo na ordem que funciona. A ordem importa por
três motivos, registrados no próprio script:

- o PyTorch vem do índice com CUDA 12.8, que o `pip install` comum não usa;
- o MMCV, o chumpy e o xtcocotools têm `setup.py` antigos e precisam de
  `setuptools<81`, sem isolamento de build;
- o MMCV entra como `mmcv-lite`, sem operadores compilados, e o script instala o
  substituto de `mmcv._ext` de que o MMPose precisa para carregar
  (`src/models/mmcv_ext_stub.py`).

A última linha deve dizer `CUDA disponível: True`. Se disser `False`, o driver
está abaixo de 570 ou o PyTorch veio do índice errado.

### 4. Baixar os pesos

Os pesos não estão no git: somam 1,1 GB.

```bash
python scripts/baixar_pesos.py
```

O script baixa cinco arquivos e confere cada um pelo SHA-256:

| Arquivo | Origem | Tamanho |
|---|---|---|
| Detector de pessoas YOLO26n-pose | release oficial da Ultralytics | 7 MB |
| Estimador 2D da montagem de mesa (RTMW-x adaptado ao cinza) | release `pesos-v1` deste repositório | 353 MB |
| Estimador 2D da montagem de retrovisor (RTMW-x adaptado ao infravermelho) | release `pesos-v1` | 353 MB |
| Lifting 3D da mesa (DSTFormer, corte de quadro) | release `pesos-v1` | 166 MB |
| Lifting 3D do retrovisor (DSTFormer, adaptação veicular) | release `pesos-v1` | 165 MB |

Se os pesos chegaram por outro meio (Drive, pendrive), aponte a pasta:

```bash
python scripts/baixar_pesos.py --origem ~/Downloads/pesos
```

### 5. Conferir a instalação

```bash
for t in tests/test_*.py; do python "$t" || echo "FALHOU: $t"; done
```

Os dez arquivos rodam em menos de um minuto, e nenhum pode imprimir `FALHOU`.
Um deles, `test_sequence_lifter.py`, confere o lifting contra janelas do H3WB e
exige o conjunto, o checkpoint do treino base e a GPU; sem eles imprime
`PULADO` e não conta como falha. O MMCV imprime dezenas de avisos
`RuntimeWarning` sobre operadores ausentes: são esperados e inofensivos.

### 6. Rodar o painel

```bash
python scripts/run_panel.py
```

Abre uma janela com quatro quadrantes: a imagem com os pontos 2D, o esqueleto
3D, a confiança por região e a latência de cada estágio. Teclas: espaço pausa,
`r` grava, `s` salva o quadro, `k` liga e desliga o esqueleto, `q` sai.

Posicione-se a cerca de um metro da câmera, com o tronco em quadro. O esqueleto
3D desenha o corpo inteiro: traço cheio é o que a câmera observou, traço fino é
previsão do modelo, o que é normal para as pernas numa webcam de mesa.

**Duas coisas mudam de uma webcam para outra**, e sem elas o esqueleto sai na
escala errada (a forma continua certa):

1. **A calibração.** `configs/camera/webcam.calibration.json` é a da webcam do
   autor (Logitech C922). Para calibrar a sua, imprima o tabuleiro em escala
   100% e siga as instruções do script:

   ```bash
   python scripts/calibrate_camera.py --gerar-tabuleiro tabuleiro.pdf
   python scripts/calibrate_camera.py --lado-quadrado 0.025
   ```

   Ou rode sem calibração, com a escala apenas aproximada:
   `python scripts/run_panel.py --calibracao ""`.

2. **A distância até a câmera**, em metros, que uma câmera só não mede:
   `python scripts/run_panel.py --distancia 0.9`. O padrão é 1,17 m.

**Não rode o painel com um treino em andamento.** A disputa pela GPU derruba a
taxa para cerca de um terço da real, e o número exibido induz a erro.

### Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `CUDA disponível: False` | Driver abaixo de 570; atualize o driver NVIDIA |
| Painel abaixo de 15 FPS com a GPU livre | A webcam entregando poucos quadros em pouca luz; o painel tenta desligar a taxa dinâmica pelo `v4l2-ctl` (`sudo apt install v4l-utils`) |
| `Nao foi possivel abrir a fonte: 0` | Outra câmera no índice 0; tente `--source 1` |
| `SHA-256 não confere` | Download interrompido; rode `baixar_pesos.py` de novo |
| `No module named 'mmcv._ext'` | O passo 5 do `instalar_ambiente.sh` não rodou; rode o script de novo |

---

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

---

## Reproduzir as medições e os treinos

Isto vai além de testar o painel, e exige os conjuntos de dados, que não podem
ser redistribuídos: cada um se obtém no site do autor, e os do Drive&Act e do
H3WB pedem cadastro e aceite de licença.

| Conjunto | Uso no projeto | Onde fica |
|---|---|---|
| COCO-WholeBody (imagens do COCO 2017 + anotações whole-body) | Adaptação ao cinza, whole-body AP | `data/raw/` → `data/processed/grayscale/` |
| Drive&Act, vídeos NIR do retrovisor e pose 3D OpenPose | Adaptação ao infravermelho, erro corporal veicular, lifting veicular | `~/Downloads/extracted/` → `data/processed/driveact/` |
| H3WB (133 keypoints 3D sobre o Human3.6M) | Treino e avaliação do lifting | `data/raw/h3wb/` → `data/processed/h3wb/` |

Conversões:

```bash
python src/data/convert_to_gray.py --src data/raw --dst data/processed/grayscale
python scripts/convert_driveact.py --videos ~/Downloads/extracted/inner_mirror \
    --poses ~/Downloads/extracted/openpose_3d \
    --splits ~/Downloads/extracted/activities_3s/inner_mirror \
    --split-name midlevel.chunks_90.split_0 --output data/processed/driveact \
    --subset val --frame-stride 5
python scripts/convert_h3wb.py --source data/raw/h3wb/reformatado \
    --output data/processed/h3wb/h3wb_annotations.npz
```

Com os dados no lugar:

```bash
python scripts/baixar_pesos.py --todos      # inclui o RTMW-x original, base dos treinos
python scripts/run_validation_battery.py    # os oito critérios do Projeto Físico

# O mesmo painel sobre um vídeo do Drive&Act, na montagem de retrovisor
python scripts/run_panel.py --source <video>.mp4 --montagem retrovisor \
    --calibracao <video>.calibration.json

# Avaliar um checkpoint 2D
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
    --ckpt <checkpoint> --data-root data/processed/grayscale/ --tag <nome>

# Treinar: um config por treino em configs/, com o motivo de cada escolha
python scripts/train_wholebody.py --config configs/rtmw_x_wholebody_gray_lora.py
```

Os `scripts/run_*.sh` são as filas que produziram as medições dos documentos.
Ficam como registro do comando exato de cada número, e esperam a GPU livre antes
de começar.

---

## Estrutura

```
configs/      Configs do MMPose: avaliação, adaptação de domínio, lifting
scripts/      Controllers — leem argumentos, chamam o modelo, devolvem a view
src/models/   Configuração de operação, pipeline de inferência, LoRA, lifting
src/data/     Datasets e conversores (Drive&Act, H3WB), calibração
src/evaluation/  Métricas e protocolos de medição
src/visualization/  Painel e desenho de esqueleto
tests/        Verificações do que falha em silêncio
results/      Medições versionadas, com o config que produziu cada uma
```

`CLAUDE.md` é a documentação corrente: ambiente, armadilhas conhecidas do
MMPose, números medidos e as regras de arquitetura do projeto. Leia-o antes de
mexer no código.
