# Real-Time 2D Full-Body Pose Estimation for Grayscale (Infrared) Images

Sistema de estimação de pose 2D full-body (133 keypoints) em imagens grayscale/infrared em tempo real para aplicações veiculares.

**Status**: Treinamento completo | AP 0.4373 | Checkpoint: `work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth`

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Troubleshooting](#troubleshooting)
- [Resultados](#resultados)
- [Referências](#referências)

---

## Visão Geral

### Objetivo

Desenvolver um sistema de estimação de pose 2D full-body em imagens grayscale/infrared em tempo real, focando em aplicações de monitoramento veicular (motoristas e passageiros).

### Por que Grayscale/Infrared?

- **Privacidade**: Não captura cores/detalhes faciais sensíveis
- **Robustez**: Funciona em baixa luminosidade e à noite
- **Aplicação prática**: Câmeras IR são padrão em sistemas de monitoramento interno de veículos modernos

### Arquitetura: Top-Down Approach

**RTMPose** com abordagem top-down para máxima precisão:

1. **RTMDet** (opcional): Detecta pessoas na imagem → bounding boxes
2. **RTMPose**: Estima pose de cada pessoa → 133 keypoints

**Alternativa bottom-up**: Detecta todos keypoints primeiro, depois agrupa em pessoas (menos preciso, mas mais rápido para multidões).

### Pipeline

\`\`\`
COCO-WholeBody (RGB) → Conversão Grayscale + Simulação IR → 
Data Augmentation → Fine-tuning RTMPose → Avaliação
\`\`\`

---

## Dataset

### COCO-WholeBody

Dataset principal utilizado para treinamento e avaliação.

- **Fonte**: [COCO-WholeBody GitHub](https://github.com/jin-s13/COCO-WholeBody)
- **Tamanho**: 
  - Training: ~118,000 imagens
  - Validation: ~5,000 imagens
  
#### 133 Keypoints por Pessoa

| Região | Keypoints | Descrição |
|--------|-----------|-----------|
| **Body** | 17 | nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles |
| **Face** | 68 | facial landmarks completos |
| **Hands** | 42 | 21 pontos por mão (dedos, palma, pulso) |
| **Feet** | 6 | pontos de apoio dos pés |

**Formato**: Anotações COCO JSON com coordenadas (x, y) e flag de visibilidade para cada keypoint.

### Conversão RGB → Grayscale

Para simular câmeras infravermelhas:

\`\`\`python
Gray = 0.299*R + 0.587*G + 0.114*B
\`\`\`

Fórmula ponderada que preserva a luminância percebida e simula melhor sensores IR.

---

## Data Augmentation

Técnicas aplicadas para aumentar robustez em condições reais de câmeras IR:

### 1. Vignetting (Simulação de Lentes IR)
Escurece as bordas da imagem (~30% de redução) para simular efeito de vinheta comum em lentes IR.

**Motivação**: Câmeras IR frequentemente apresentam perda de intensidade nas extremidades.

### 2. Ruído Gaussiano
Adiciona ruído aleatório (~1% de intensidade) para simular ruído térmico de sensores IR.

**Motivação**: Sensores IR têm características de ruído diferentes de câmeras RGB tradicionais.

### 3. Gaussian Blur
Desfoque suave (kernel 3×3 ou 5×5) para simular limitações ópticas.

**Motivação**: Câmeras IR podem ter menor resolução ou abertura limitada.

### 4. Ajustes de Contraste
Variação aleatória de ±20% no contraste.

**Motivação**: Iluminação IR varia com fonte e distância ao objeto.

### 5. Ajustes de Brilho
Variação aleatória de ±30 pontos no brilho global.

**Motivação**: Diferentes materiais refletem IR de forma distinta.

### 6. Rotação e Flip
Rotação aleatória (±15°) e flip horizontal (50% chance).

**Motivação**: Aumenta invariância rotacional e bilateral do modelo.

### Pipeline de Augmentation

\`\`\`python
RGB → Grayscale → [Vignetting] → [Noise] → [Blur] → 
[Contrast/Brightness] → [Rotation/Flip] → Training
\`\`\`

Augmentações entre colchetes são aplicadas aleatoriamente durante o treinamento.

---

## Instalação

### Requisitos

- **OS**: Linux (testado em Ubuntu)
- **GPU**: NVIDIA com driver/CUDA (testado RTX 5060, CUDA 12.6)
- **Python**: 3.10–3.12
- **Espaço**: ~100GB (datasets + checkpoints)

### Passo a Passo

\`\`\`bash
# 1. Entre no diretório do projeto
cd /home/davs/Documents/TCC/Project

# 2. Crie e ative ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 3. Atualize pip
pip install --upgrade pip

# 4. Instale PyTorch com CUDA 12.6 (ajuste conforme sua CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5. Instale OpenMMLab via openmim
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
mim install mmpose

# 6. Instale dependências auxiliares
pip install -r requirements.txt

# 7. Verifique instalação
python -c "import torch, mmcv, mmdet, mmpose; print('✅ OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
\`\`\`

**Nota**: Ajuste a URL do PyTorch conforme sua versão CUDA em https://pytorch.org/get-started/locally/

---

## Como Usar

### 1. Preparar Dataset

\`\`\`bash
# Automático (recomendado)
bash scripts/prepare_dataset.sh

# OU manual:
python src/data/download_coco.py  # Download COCO-WholeBody
python src/data/convert_to_gray.py \\
  --input-dir data/raw \\
  --output-dir data/processed/grayscale \\
  --simulate-ir \\
  --apply-augmentation
\`\`\`

**Resultado**: Estrutura em \`data/processed/grayscale/{train2017,val2017,annotations}/\`

### 2. Treinar Modelo

\`\`\`bash
# Fine-tuning completo (recomendado)
python src/training/train_pose.py \\
  --config configs/rtmpose_m_wholebody.py \\
  --load-from checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \\
  --work-dir work_dirs/finetune_grayscale \\
  --amp

# OU teste rápido (1 epoch, dry-run)
python src/training/train_pose.py \\
  --config configs/rtmpose_m_wholebody_minimal.py \\
  --work-dir work_dirs/test_minimal
\`\`\`

**Checkpoints pré-treinados**: Coloque em \`checkpoints/\` (veja seção Referências).

### 3. Avaliar Modelo

\`\`\`bash
# Comparar RGB vs Grayscale
python src/evaluation/evaluate_pose.py \\
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \\
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
  --rgb-dir data/raw/val2017 \\
  --ir-dir data/processed/grayscale/val2017 \\
  --out-dir work_dirs/eval_results \\
  --n 20
\`\`\`

**Saída**: Imagens anotadas e métricas em \`work_dirs/eval_results/{rgb,ir}/\`

### 4. Inferência em Tempo Real

#### Webcam

\`\`\`bash
python src/evaluation/run_realtime.py \\
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \\
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
  --device cuda:0 \\
  --source 0
\`\`\`

#### Vídeo

\`\`\`bash
python src/evaluation/run_realtime.py \\
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \\
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
  --device cuda:0 \\
  --source data/video/seu_video.mp4
\`\`\`

#### Com Detector de Pessoas (Multi-Pessoa)

\`\`\`bash
python src/evaluation/run_realtime.py \\
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \\
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \\
  --det-cfg configs/detectors/rtmdet_nano_person_infer.py \\
  --det-ckpt checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \\
  --bbox-thr 0.5 \\
  --score-thr 0.4 \\
  --device cuda:0 \\
  --source 0
\`\`\`

**Controles**: Pressione \`q\` para sair | FPS exibido no canto superior esquerdo

### 5. Visualizar Treinamento

\`\`\`bash
# TensorBoard
tensorboard --logdir work_dirs/test_minimal5 --port 6006
# Abra http://localhost:6006

# OU script de plotagem
python scripts/plot_training.py \\
  --logdir work_dirs/test_minimal5 \\
  --out plots/training_curves.png
\`\`\`

---

## Estrutura do Projeto

\`\`\`
Project/
├── data/
│   ├── raw/                          # COCO-WholeBody original (RGB)
│   └── processed/grayscale/          # Convertido para grayscale + augmentation
│
├── src/
│   ├── data/
│   │   ├── download_coco.py          # Download do dataset
│   │   ├── convert_to_gray.py        # Conversão RGB → Grayscale
│   │   └── augmentation.py           # Data augmentation
│   ├── training/
│   │   └── train_pose.py             # Script de treinamento
│   └── evaluation/
│       ├── evaluate_pose.py          # Avaliação RGB vs Grayscale
│       └── run_realtime.py           # Inferência em tempo real
│
├── configs/
│   ├── rtmpose_m_wholebody.py        # Config principal (completa)
│   ├── rtmpose_m_wholebody_minimal.py # Config rápida (testes)
│   └── detectors/
│       └── rtmdet_nano_person_infer.py # Config RTMDet
│
├── checkpoints/                      # Checkpoints pré-treinados
├── work_dirs/                        # Outputs de treinamento
│   ├── test_minimal5/                # ⭐ Melhor modelo atual
│   │   └── best_coco-wholebody_AP_epoch_50.pth
│   └── eval_results/                 # Resultados de avaliação
│
├── scripts/
│   ├── prepare_dataset.sh            # Preparação automática
│   ├── plot_training.py              # Visualização de curvas
│   └── train_full_pipeline.sh        # Pipeline completo
│
├── requirements.txt
└── README.md
\`\`\`

---

## Troubleshooting

### Erro: \`ModuleNotFoundError: No module named 'mmcv._ext'\`

**Causa**: MMCV precisa de extensões compiladas (ops CUDA).

**Solução**:
\`\`\`bash
mim install "mmcv>=2.0.0,<2.2.0"
\`\`\`

### Erro: Incompatibilidade de versões (mmcv, mmdet, mmpose)

**Solução**: Reinstalar com versões compatíveis:
\`\`\`bash
pip uninstall mmcv mmdet mmpose mmengine -y
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
mim install mmpose
\`\`\`

### Erro: CUDA Out of Memory

**Soluções**:
- Reduzir \`batch_size\` no config (32 → 16 → 8)
- Usar flag \`--amp\` (mixed precision)
- Usar config minimal em vez de completa

### Erro: PyTorch 2.6+ Weights Only Load Failed

**Causa**: PyTorch 2.6+ mudou padrão de \`weights_only\` para \`True\`.

**Solução**: Já corrigido no código (\`run_realtime.py\` faz monkey-patch do \`torch.load\`).

### Aviso: GPU não compatível (RTX série 50 / Ada Lovelace)

**Solução**: Instalar PyTorch com suporte CUDA 12.x:
\`\`\`bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
\`\`\`

---

## Resultados

### Status Atual

- **Modelo**: RTMPose-M fine-tuned para grayscale
- **Treinamento**: 50 epochs (~24h em RTX 5060 8GB)
- **AP (COCO-WholeBody val)**: 0.4373
- **Checkpoint**: \`work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth\`

### Performance Esperada (Fine-tuning Completo)

| Métrica | Valor Alvo | Observação |
|---------|------------|------------|
| **AP (Grayscale)** | 60-65% | Com treinamento completo |
| **FPS** | 35-50 | Single person, RTX 5060 |
| **Latência** | <30ms | Por frame |
| **VRAM** | ~4-6GB | Permite batch inference |
| **Degradação RGB→IR** | <10% | Meta do projeto |

### Métricas COCO

| Métrica | Descrição | Threshold |
|---------|-----------|-----------|
| **AP** | Average Precision | OKS 0.5:0.95 |
| **AP@0.5** | AP em OKS 0.5 | OKS 0.5 |
| **AP@0.75** | AP em OKS 0.75 | OKS 0.75 |
| **PCK** | Percentage Correct Keypoints | 0.2 × torso |
| **AR** | Average Recall | OKS 0.5:0.95 |

**OKS** (Object Keypoint Similarity): Métrica padrão COCO, similar ao IoU mas para keypoints.

---

## Referências

### Papers

- [RTMPose: Real-Time Multi-Person Pose Estimation](https://arxiv.org/abs/2303.07399) (2023)
- [COCO-WholeBody: COCO with Whole-Body Keypoint Annotations](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12) (ECCV 2020)
- [OpenPose: Realtime Multi-Person 2D Pose Estimation](https://arxiv.org/abs/1812.08008) (2018)

### Repositórios

- [MMPose](https://github.com/open-mmlab/mmpose) - OpenMMLab Pose Estimation Toolbox
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - RTMPose implementation
- [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) - Dataset oficial

### Documentação

- [MMPose Docs](https://mmpose.readthedocs.io/)
- [MMCV Docs](https://mmcv.readthedocs.io/)
- [MMDetection Docs](https://mmdetection.readthedocs.io/)

### Checkpoints Pré-Treinados

**RTMPose-M (Body7)**:
- Link: [Checkpoint oficial](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth)
- Colocar em: \`checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth\`

**RTMDet-Nano (Person Detection)**:
- Link: [Checkpoint oficial](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)
- Colocar em: \`checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth\`

---

## Contato

**Aluno**: Davi Baechtold Campos  
**Orientador**: Prof. Dr. Alceu de Souza Brito Junior  
**Instituição**: PUCPR  
**Curso**: Engenharia de Computação  

---

## Licença

Este projeto é desenvolvido para fins acadêmicos (TCC). Bibliotecas utilizadas (MMPose, MMDetection, etc.) possuem suas próprias licenças (geralmente Apache 2.0).

---

**Última atualização**: Outubro 2025 | **Versão**: 2.0 (Simplificada)
