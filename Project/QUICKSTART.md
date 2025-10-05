# Guia de Início Rápido - Pose 2D Grayscale (Infrared)

## 📋 Pré-requisitos

### Hardware
- GPU NVIDIA com pelo menos 8GB VRAM (recomendado: RTX 3060 ou superior)
- 32GB RAM
- 200GB de espaço em disco

### Software
- Ubuntu 20.04+ ou Windows 10/11 com WSL2
- Python 3.8+
- CUDA 12.6+ (driver NVIDIA 560+ recomendado)
- Git

## 🚀 Instalação

### 1. Clonar o repositório
```bash
cd /home/davs/Documents/TCC/Project
```

### 2. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências
```bash
# Upgrade pip
pip install --upgrade pip

# Instalar PyTorch com CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Instalar outras dependências
pip install -r requirements.txt

# Instalar MMPose e MMDetection
pip install -U openmim
mim install mmengine mmcv mmdet mmpose

# Verificar instalação CUDA + PyTorch
python - << 'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
PY
```

## 📊 Preparação dos Dados

### Passo 1: Download automático do COCO
```bash
# Dar permissão de execução ao script
chmod +x scripts/prepare_dataset.sh

# Executar preparação
bash scripts/prepare_dataset.sh
```

Este script irá:
1. Baixar imagens do COCO-WholeBody (train + val)
2. Aguardar você baixar as anotações manualmente
3. Converter todas as imagens para grayscale
4. Aplicar simulações de características infrared
5. Gerar visualizações

### Passo 2: Download manual das anotações
Durante a execução do script, você será solicitado a baixar:

**Training annotations:**
- Link: https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf
- Salvar como: `data/raw/annotations/coco_wholebody_train_v1.0.json`

**Validation annotations:**
- Link: https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgGk0dCTj8xxt
- Salvar como: `data/raw/annotations/coco_wholebody_val_v1.0.json`

### Passo 3: Verificar dataset
```bash
python src/data/download_coco.py --data-dir data/processed/grayscale --verify-only
```

## 🎯 Treinamento

### Baseline: RGB para Grayscale
```bash
# Treinar com configuração padrão
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/baseline_gray
```

### Com Fine-tuning
```bash
# Baixar checkpoint pré-treinado do RTMPose
mkdir -p checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
cd ..

# Fine-tune a partir do checkpoint
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --load-from checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --work-dir work_dirs/finetune_gray
```

### Com Mixed Precision (AMP)
```bash
# Treinar com FP16 para acelerar
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/amp_gray \
    --amp
```

### Multi-GPU
```bash
# Treinar com 2 GPUs
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/multigpu_gray \
    --gpu-ids 0 1
```

## 📈 Monitoramento

### TensorBoard
```bash
# Em um terminal separado
tensorboard --logdir work_dirs/
```

Acesse: http://localhost:6006

### Weights & Biases (opcional)
```bash
# Login no W&B
wandb login

# W&B será automaticamente detectado pelo MMPose
```

## 🧪 Avaliação

### Avaliar modelo treinado
```bash
python src/evaluation/evaluate.py \
    --config configs/rtmpose_m_wholebody.py \
    --checkpoint work_dirs/baseline_gray/latest.pth \
    --work-dir work_dirs/baseline_gray/eval
```

### Visualizar predições
```bash
python src/evaluation/visualize.py \
    --config configs/rtmpose_m_wholebody.py \
    --checkpoint work_dirs/baseline_gray/latest.pth \
    --img-dir data/processed/grayscale/val2017 \
    --output-dir work_dirs/baseline_gray/visualizations \
    --num-samples 50
```

## 📊 Estrutura de Diretórios Após Setup

```
Project/
├── data/
│   ├── raw/                          # Dados RGB originais
│   │   ├── train2017/               # ~118K imagens
│   │   ├── val2017/                 # ~5K imagens
│   │   └── annotations/
│   │       ├── coco_wholebody_train_v1.0.json
│   │       └── coco_wholebody_val_v1.0.json
│   └── processed/
│       └── grayscale/               # Dados convertidos
│           ├── train2017/
│           ├── val2017/
│           └── annotations/
│
├── work_dirs/                        # Outputs de treinamento
│   └── baseline_gray/
│       ├── *.pth                    # Checkpoints
│       ├── *.log                    # Logs
│       └── tf_logs/                 # TensorBoard
│
├── checkpoints/                      # Modelos pré-treinados
│   └── rtmpose-m_*.pth
│
└── src/
    ├── data/                         # Scripts de dados
    ├── training/                     # Scripts de treino
    └── evaluation/                   # Scripts de avaliação
```

## 🔍 Troubleshooting

### Erro: CUDA out of memory
**Solução:** Reduzir batch size no config
```python
# Em configs/rtmpose_m_wholebody.py
train_dataloader = dict(
    batch_size=32,  # Reduzir de 64 para 32 ou 16
    ...
)
```

### Erro: Dataset not found
**Solução:** Verificar estrutura de diretórios
```bash
python src/data/download_coco.py --verify-only
```

### Erro: Import error (cv2, torch, etc)
**Solução:** Reinstalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Treinamento muito lento
**Soluções:**
1. Usar AMP (--amp flag)
2. Reduzir num_workers
3. Usar GPUs mais potentes
4. Reduzir image_size

## 📚 Próximos Passos

1. **Exploração de Dados**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Testar Augmentations**
   ```bash
   jupyter notebook notebooks/02_augmentation_tests.ipynb
   ```

3. **Análise de Resultados**
   ```bash
   jupyter notebook notebooks/03_model_evaluation.ipynb
   ```

## 🎓 Conceitos Importantes

### Bottom-Up vs Top-Down
- **Bottom-Up**: Detecta todos os keypoints primeiro, depois agrupa
  - Exemplo: OpenPose
  - Vantagem: Mais rápido com muitas pessoas
  - Desvantagem: Menos preciso

- **Top-Down** (usado neste projeto):
  - Detecta pessoas primeiro, depois estima pose de cada uma
  - Exemplo: RTMDet + RTMPose
  - Vantagem: Mais preciso
  - Desvantagem: Tempo cresce com número de pessoas

### COCO-WholeBody Keypoints
- Body: 17 keypoints (padrão COCO)
- Face: 68 keypoints
- Left Hand: 21 keypoints
- Right Hand: 21 keypoints
- Feet: 6 keypoints
- **Total: 133 keypoints**

### Métricas de Avaliação
- **OKS (Object Keypoint Similarity)**: Similar ao IoU para keypoints
- **AP (Average Precision)**: Precisão média em diferentes thresholds
- **AR (Average Recall)**: Recall médio
- **PCK (Percentage of Correct Keypoints)**: % de keypoints corretos

## 📖 Referências

- RTMPose Paper: https://arxiv.org/abs/2303.07399
- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody
- MMPose: https://github.com/open-mmlab/mmpose

## 🤝 Suporte

Para dúvidas ou problemas:
1. Verificar [Issues do MMPose](https://github.com/open-mmlab/mmpose/issues)
2. Contatar o orientador: Prof. Dr. Alceu de Souza Brito Junior
3. Email: davi.baechtold@pucpr.br
