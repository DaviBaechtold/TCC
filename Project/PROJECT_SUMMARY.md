# Resumo do Projeto: Pose 2D Full-Body para Imagens Grayscale

## 🎯 Objetivo

Construir e treinar uma rede neural para detecção de pose 2D full-body em **imagens grayscale (infrared)** em tempo real, focando em aplicações para ambientes veiculares.

## 🏗️ Arquitetura

### Abordagem: Top-Down
1. **RTMDet** detecta pessoas (bounding boxes)
2. **RTMPose** estima pose de cada pessoa detectada

### Modelo: RTMPose-m (Medium)
- **Backbone**: CSPNeXt
- **Head**: RTMCCHead (SimCC approach)
- **Keypoints**: 133 (WholeBody)
  - Body: 17
  - Face: 68
  - Hands: 42 (21 cada)
  - Feet: 6

## 📊 Dataset: COCO-WholeBody

### Características
- **Training**: ~118,000 imagens
- **Validation**: ~5,000 imagens
- **Anotações**: 133 keypoints por pessoa
- **Link**: https://github.com/jin-s13/COCO-WholeBody

### Conversão para Grayscale
Todas as imagens RGB são convertidas para grayscale com simulação de características infrared:
- Conversão luminosity-based (0.299R + 0.587G + 0.114B)
- Adição de ruído térmico
- Vignetting
- Hot pixels

## 🔄 Pipeline Completo

```
1. Download COCO-WholeBody (RGB)
   ↓
2. Conversão RGB → Grayscale
   ↓
3. Simulação de características IR
   ↓
4. Data Augmentation
   ↓
5. Treinamento RTMPose
   ↓
6. Avaliação e Fine-tuning
   ↓
7. Otimização para tempo real
```

## 💾 Data Augmentation

### Geométricas
- Horizontal Flip (50%)
- Rotation (-30° a +30°)
- Scale (0.6x a 1.4x)
- Shift
- Perspective transform

### Fotométricas
- Brightness/Contrast
- Gamma correction
- CLAHE (Contrast Limited AHE)

### Específicas para IR
- **Vignetting** (escurecimento nas bordas)
- **Ruído térmico** (gaussiano)
- **Hot pixels** (pixels defeituosos)
- Blur (Gaussian, Motion, Median)

## 🎓 Conceitos Chave

### Bottom-Up vs Top-Down

| Aspecto | Bottom-Up | Top-Down |
|---------|-----------|----------|
| **Ordem** | Keypoints → Pessoas | Pessoas → Keypoints |
| **Precisão** | Menor | Maior |
| **Velocidade** | Mais rápido (muitas pessoas) | Depende do número de pessoas |
| **Exemplo** | OpenPose | RTMDet + RTMPose |
| **Uso** | Crowds, múltiplas pessoas | Alta precisão, poucas pessoas |

### Escolha: Top-Down
- Ambiente veicular típico: 1-4 pessoas
- Foco em precisão sobre velocidade bruta
- RTMPose já é otimizado para tempo real

## 📈 Métricas de Avaliação

### Object Keypoint Similarity (OKS)
$$
OKS = \frac{\sum_i \exp(-d_i^2 / 2s^2\sigma_i^2) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}
$$

Onde:
- $d_i$: distância euclidiana entre predição e ground truth
- $s$: escala do objeto (área)
- $\sigma_i$: desvio padrão por keypoint
- $v_i$: visibilidade do keypoint

### Average Precision (AP)
- **AP@0.5**: IoU threshold = 0.5
- **AP@0.75**: IoU threshold = 0.75
- **AP@0.5:0.95**: Média de 0.5 a 0.95 (step 0.05)

### Métricas Específicas
- **AP (Body)**: Para 17 keypoints do corpo
- **AP (Face)**: Para 68 keypoints faciais
- **AP (Hand)**: Para 42 keypoints das mãos
- **AP (Foot)**: Para 6 keypoints dos pés

### Performance
- **FPS**: Frames per second
- **Latência**: ms por frame
- **Throughput**: imagens/segundo

## 🔧 Configuração de Treinamento

### Hiperparâmetros
```python
batch_size = 64           # Training
batch_size = 32           # Validation
num_workers = 8           # Data loading
max_epochs = 420
base_lr = 4e-3
weight_decay = 0.05
optimizer = AdamW
```

### Learning Rate Schedule
1. **Warmup** (Linear): 0-1000 steps
2. **Cosine Annealing**: 150-420 epochs

### Input Size
- **Image**: 256 x 192 (H x W)
- **Output**: 133 keypoints (x, y, visibility)

## 📂 Estrutura de Arquivos

```
Project/
├── README.md                 # Overview do projeto
├── QUICKSTART.md            # Guia de início rápido
├── requirements.txt         # Dependências Python
│
├── data/
│   ├── raw/                 # COCO RGB original
│   └── processed/           # Grayscale convertido
│
├── src/
│   ├── data/
│   │   ├── download_coco.py       # Download dataset
│   │   ├── convert_to_gray.py     # RGB → Gray
│   │   └── augmentation.py        # Data augmentation
│   ├── training/
│   │   └── train_pose.py          # Script de treino
│   └── evaluation/
│       ├── metrics.py             # Cálculo de métricas
│       └── visualize.py           # Visualizações
│
├── configs/
│   └── rtmpose_m_wholebody.py    # Config do modelo
│
└── scripts/
    └── prepare_dataset.sh         # Preparação automática
```

## 🚀 Comandos Rápidos

### Preparar Dataset
```bash
bash scripts/prepare_dataset.sh
```

### Treinar Modelo
```bash
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/baseline
```

### Avaliar Modelo
```bash
python src/evaluation/evaluate.py \
    --config configs/rtmpose_m_wholebody.py \
    --checkpoint work_dirs/baseline/latest.pth
```

## 📊 Resultados Esperados

### Baseline (RGB)
- **AP**: ~65-70%
- **AP (Body)**: ~75%
- **AP (Face)**: ~65%
- **AP (Hand)**: ~55%
- **FPS**: ~30-40 (RTX 3060)

### Target (Grayscale - RTX 5060)
- **AP**: ~60-65% (objetivo: <5% drop vs RGB)
- **FPS**: ~35-50 (melhor que RTX 3060)
- **Latência**: <30ms
- **Throughput**: ~40-50 img/s em batch inference

## 🎯 Critérios de Sucesso

1. ✅ **Precisão**: AP > 60% em grayscale
2. ✅ **Tempo Real**: FPS ≥ 30 em GPU mid-range
3. ✅ **Robustez**: <10% degradação com ruído
4. ✅ **Generalização**: Funciona em ambiente veicular

## 📚 Próximos Passos

### Fase 1: Baseline (2 semanas)
- [x] Setup do projeto
- [ ] Download e preparação de dados
- [ ] Treinamento baseline
- [ ] Avaliação inicial

### Fase 2: Grayscale (3 semanas)
- [ ] Conversão para grayscale
- [ ] Fine-tuning para grayscale
- [ ] Data augmentation IR
- [ ] Avaliação comparativa

### Fase 3: Otimização (2 semanas)
- [ ] Quantização INT8
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Benchmark tempo real

### Fase 4: Validação (2 semanas)
- [ ] Teste em ambiente veicular
- [ ] Análise de casos de falha
- [ ] Documentação final
- [ ] Artigo científico

## 📖 Referências

### Papers
1. **RTMPose**: Jiang et al., 2023 - arXiv:2303.07399
2. **COCO-WholeBody**: Jin et al., 2020 - ECCV
3. **SimCC**: Li et al., 2022 - ECCV

### Repos
- MMPose: https://github.com/open-mmlab/mmpose
- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody

### Docs
- MMPose Docs: https://mmpose.readthedocs.io/
- PyTorch: https://pytorch.org/docs/

## 👤 Autor

**Davi Baechtold Campos**
- Instituição: PUCPR
- Curso: Engenharia de Computação
- Orientador: Prof. Dr. Alceu de Souza Brito Junior
- Email: davi.baechtold@pucpr.br

---

**Data de criação**: Outubro 2025
**Última atualização**: Outubro 2025
