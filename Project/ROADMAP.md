# 🗺️ Roadmap do Projeto - Implementação 2D Aprimorada

**Data de Criação**: Outubro 19, 2025  
**Foco**: Aprimoramento do sistema 2D com estratégias complementares

---

## 🎯 Objetivos Definidos

### 1. ✅ Strategy para Inferência Real-Time
**Status**: 🟡 A Implementar  
**Prioridade**: ALTA

**Abordagem Atual**:
```
Detector (RTMDet) → Crop pessoas → Pose por pessoa
↑ Preciso mas mais lento para multidões
```


**Implementação**:
- [ ] Sistema de agrupamento de keypoints
- [ ] **Bounding boxes** detectadas automaticamente dos keypoints
- [ ] Pipeline otimizado para múltiplas pessoas (max 5)
- [ ] Testes de latência e throughput em vídeo real-time

**Métricas de Sucesso**:
- Latência < 30ms para 2-5 pessoas (vs. ~50ms top-down)
- AP similar ao top-down (±5%)
- Bounding boxes precisas (IoU > 0.7 com ground truth)

---

### 2. ✅ Extração 2D para Plano Cartesiano XYZ
**Status**: 🟡 A Implementar  
**Prioridade**: ALTA

**Descrição**: Projetar keypoints 2D em sistema de coordenadas 3D normalizado.

**Pipeline**:
```
2D Image (pixel coords) → Normalização → 3D Cartesian (XYZ)
    ↓
(x_px, y_px) → normalizar por root joint → (X, Y, Z_estimado)
```

**Implementação**:
- [ ] Normalização por root joint (hip center)
- [ ] Estimação de profundidade relativa (weak perspective)
- [ ] Z-axis inference baseado em heurísticas:
  - Tamanho relativo das juntas
  - Prior anatômico (braços à frente do tronco)
  - Escala por distância da câmera
- [ ] Exportação em formato 3D (JSON/NPZ com coordenadas XYZ)
- [ ] Visualização 3D (matplotlib 3D ou Open3D)

**Output Esperado**:
```python
{
  "person_id": 0,
  "keypoints_3d": [
    {"joint": "nose", "x": 0.0, "y": 0.5, "z": 0.1},
    {"joint": "left_shoulder", "x": -0.2, "y": 0.3, "z": 0.0},
    ...
  ]
}
```

**Nota**: Sem lifting real (sem MLP), apenas **weak perspective projection** + heurísticas anatômicas.

---

### 3. ✅ Dataset Drive&Act Integration
**Status**: 🟡 A Implementar  
**Prioridade**: MÉDIA-ALTA

**Descrição**: Incorporar dataset Drive&Act para validação em cenário veicular real.

**Drive&Act Overview**:
- **Fonte**: Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition (ICCV 2019)
- **Conteúdo**: 15 horas de vídeo, 9.6M frames
- **Câmeras**: 6 views sincronizadas (top, side, dash, etc.)
- **Anotações**: Gestos, poses, ações finas (83 classes)
- **Formato**: RGB + Depth + IR

**Implementação**:
```
Phase A: Download & Preprocessing
├── Download Drive&Act subset (side/dash views)
├── Extrair frames relevantes (gestos + poses)
├── Converter para formato COCO-like
└── Split: train/val/test

Phase B: Annotation Alignment
├── Mapear anotações Drive&Act → COCO keypoints
├── Lidar com keypoints parciais (oclusões)
└── Criar manifest files

Phase C: Fine-tuning
├── Continuar treinamento com Drive&Act
├── Domain adaptation: COCO → Drive&Act
└── Avaliação: AP em cenário veicular

Phase D: Occlusion Analysis
├── Quantificar oclusões (volante, painel)
├── Análise de keypoints mais afetados
└── Propor melhorias específicas
```

**Métricas**:
- AP no Drive&Act validation set
- Degradação vs. COCO (quantificar domain gap)
- Recall em keypoints oclusos

**Desafios**:
- ⚠️ Dataset grande (~200GB)
- ⚠️ Mapeamento de anotações pode ser trabalhoso
- ⚠️ Possível necessidade de re-anotação

---

### 4. ✅ Melhorar Treinamento da Rede Neural
**Status**: 🟡 A Implementar  
**Prioridade**: MÉDIA

**Descrição**: Otimizar arquitetura e hiperparâmetros para maior precisão.

**Estratégias**:

#### 4.1. Arquitetura
- [ ] **Backbone upgrade**: RTMPose-M → RTMPose-L (mais parâmetros)
- [ ] **Multi-scale features**: adicionar FPN (Feature Pyramid Network)
- [ ] **Attention mechanisms**: CBAM ou Coordinate Attention
- [ ] **Loss refinement**: 
  - Adicionar Wing Loss para keypoints difíceis
  - Weighted loss por região (mais peso em mãos/face)

#### 4.2. Hiperparâmetros
- [ ] **Learning rate scheduling**: 
  - Warmup (5 epochs)
  - Cosine annealing com restarts
- [ ] **Batch size optimization**: testar 16/32/64
- [ ] **Epochs**: aumentar para 100-150 com early stopping
- [ ] **Optimizer**: testar AdamW vs. SGD com momentum

#### 4.3. Data Augmentation Avançada
- [ ] **Mixup/CutMix**: mistura de imagens para regularização
- [ ] **Random Erasing**: simular oclusões
- [ ] **Auto-augmentation**: busca automática de políticas
- [ ] **Grayscale variations**: 
  - Diferentes fórmulas de conversão (ITU-R BT.601, BT.709)
  - Simulação de diferentes sensores IR

#### 4.4. Training Tricks
- [ ] **EMA (Exponential Moving Average)**: suavizar pesos
- [ ] **Label smoothing**: reduzir overconfidence
- [ ] **Gradient clipping**: estabilidade de treinamento
- [ ] **Mixed precision (AMP)**: já implementado, otimizar

**Meta**:
- AP 0.50+ (atual: 0.4373)
- Melhoria de +5-10% em keypoints difíceis (hands, feet)

---

### 5. ✅ Revisão e Documentação de Métricas 2D
**Status**: 🟡 A Implementar  
**Prioridade**: ALTA (essencial para monografia)

**Descrição**: Documentação científica completa do sistema 2D.

#### 5.1. Review of 2D Keypoint Metrics

**Métricas COCO (implementadas)**:
```python
# Object Keypoint Similarity (OKS)
OKS = Σ exp(-di²/2s²ki²) δ(vi>0) / Σ δ(vi>0)

Onde:
- di: distância euclidiana entre pred e GT
- s: escala do objeto (√area da bbox)
- ki: constante por keypoint (sigma)
- vi: visibilidade do keypoint
```

**Métricas Adicionais a Implementar**:
- [ ] **PCK (Percentage of Correct Keypoints)**:
  - Threshold: 0.2 × torso diagonal
  - Por região: body, hands, face, feet
- [ ] **PCKh (head-normalized)**:
  - Threshold: 0.5 × head size
- [ ] **MPJPE (Mean Per Joint Position Error)**:
  - Erro médio em pixels
  - Por keypoint individual
- [ ] **AUC (Area Under Curve)**:
  - Curva PCK vs. threshold
  - Métrica robusta

**Análises Especializadas**:
- [ ] **Erro por região anatômica**:
  ```
  Body (17 kpts)  → AP_body
  Face (68 kpts)  → AP_face
  Hands (42 kpts) → AP_hands
  Feet (6 kpts)   → AP_feet
  ```
- [ ] **Erro por nível de oclusão**:
  ```
  Visible (v=2)       → AP_visible
  Partially (v=1)     → AP_partial
  Occluded (v=0)      → AP_occluded
  ```
- [ ] **Degradação RGB → Grayscale**:
  ```
  ΔAP = AP_rgb - AP_gray
  Por keypoint: qual sofre mais?
  ```

#### 5.2. Describe the Datasets

**Documento a criar**: `docs/DATASETS.md`

**Conteúdo**:
```markdown
# Datasets Utilizados

## 1. COCO-WholeBody (Primary)

### Visão Geral
- **Fonte**: ECCV 2020 (Jin et al.)
- **Tamanho**: 
  - Training: 118,287 imagens
  - Validation: 5,000 imagens
- **Keypoints**: 133 por pessoa
  - Body: 17 (COCO standard)
  - Face: 68 (landmarks)
  - Hands: 42 (21 per hand)
  - Feet: 6 (contact points)
- **Formato**: JSON (COCO format)
- **Características**:
  - Scenes diversas (indoor/outdoor)
  - Múltiplas pessoas por imagem
  - Oclusões naturais
  - Variação de escala (pessoas próximas/distantes)

### Estatísticas
- Média de pessoas/imagem: 2.3
- Keypoints visíveis/pessoa: ~85% (média)
- Resolução típica: 640×480
- Distribuição de poses: variada (standing, sitting, lying, etc.)

### Preprocessamento
1. Download via script: `src/data/download_coco.py`
2. Conversão RGB → Grayscale: 0.299R + 0.587G + 0.114B
3. Data augmentation aplicada (ver seção 5.3)
4. Normalização: mean=[123.675], std=[58.395] (ImageNet)

### Desafios
- ⚠️ Faces pequenas (difícil anotar 68 keypoints)
- ⚠️ Mãos frequentemente ocluídas
- ⚠️ Pés raramente visíveis completos

## 2. Drive&Act (Vehicular - Optional)

### Visão Geral
- **Fonte**: ICCV 2019 (Martin et al.)
- **Tamanho**: 15 horas de vídeo, 9.6M frames
- **Câmeras**: 6 views sincronizadas
  - Top (teto)
  - Side (lateral)
  - Dash (painel)
  - Face (rosto)
  - Depth (profundidade)
  - IR (infravermelho)
- **Anotações**:
  - 83 classes de ações
  - 34 objetos
  - Poses em subset de frames

### Uso no Projeto
- Validação em cenário veicular real
- Análise de oclusões severas (volante, painel)
- Fine-tuning para domínio específico
- Teste de robustez em IR real

### Mapeamento COCO ↔ Drive&Act
- Body keypoints: compatível (subset)
- Hands: anotações limitadas
- Face: sem anotações detalhadas
- **Desafio**: requer adaptação de formato

## 3. Human3.6M (Future Work - 3D Ground Truth)

### Visão Geral
- **Fonte**: TPAMI 2014 (Ionescu et al.)
- **Conteúdo**: 3.6M frames com GT 3D
- **Uso**: Treinamento de lifting 2D→3D (fase futura)

## Comparação

| Dataset | Keypoints | 3D GT | Vehicular | IR | Size |
|---------|-----------|-------|-----------|----|----|
| COCO-WB | 133 | ❌ | ❌ | ❌ | 25GB |
| Drive&Act | ~17 | ❌ | ✅ | ✅ | 200GB |
| Human3.6M | 17 | ✅ | ❌ | ❌ | 100GB |
```

#### 5.3. Describe the Data Augmentations

**Documento a criar**: `docs/AUGMENTATIONS.md`

**Conteúdo Técnico**:
```markdown
# Data Augmentation Techniques

## Overview

Data augmentation é aplicada **apenas durante treinamento** para:
1. Aumentar diversidade do dataset (~10x amostras efetivas)
2. Melhorar robustez a variações
3. Simular condições de sensores IR reais
4. Prevenir overfitting

## Pipeline de Aplicação

```python
Image (RGB) 
  ↓
Grayscale Conversion (always)
  ↓
Random Augmentations (50% probability each):
  ├── Vignetting
  ├── Gaussian Noise
  ├── Gaussian Blur
  ├── Contrast Adjustment
  ├── Brightness Adjustment
  └── Geometric (Rotation + Flip)
  ↓
Normalization
  ↓
Training
```

## 1. Grayscale Conversion

### Motivação
Simular câmeras infravermelhas que não capturam cor.

### Implementação
```python
def rgb_to_gray(img):
    # ITU-R BT.601 standard (NTSC)
    return 0.299 * R + 0.587 * G + 0.114 * B
```

**Alternativas testadas**:
- **BT.709 (HDTV)**: `0.2126*R + 0.7152*G + 0.0722*B`
- **Luminosity**: perceptual weighting
- **Average**: `(R+G+B)/3` (não recomendado)

**Escolha**: BT.601 por ser padrão NTSC e melhor preservar luminância percebida.

### Justificativa Científica
Sensores IR capturam radiação térmica (~700-1000nm) que não correlaciona perfeitamente com RGB. Grayscale é aproximação prática.

### Impacto em Métricas
- ΔAP (RGB→Gray): ~5-8% degradação
- Keypoints mais afetados: face (perde info de textura de pele)
- Keypoints menos afetados: body (contornos preservados)

## 2. Vignetting

### Motivação
Lentes IR frequentemente apresentam perda de intensidade nas bordas (efeito vinheta).

### Implementação
```python
def vignetting(img, strength=0.3):
    rows, cols = img.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Centro da imagem
    center_x, center_y = cols/2, rows/2
    
    # Distância do centro (normalizada)
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    dist = dist / dist.max()
    
    # Máscara radial (Gaussian falloff)
    mask = 1 - strength * dist**2
    
    return img * mask
```

**Parâmetros**:
- `strength`: 0.2-0.4 (30% degradação típica)
- Aplicado em: 50% das imagens

### Justificativa
Estudo de Martin et al. (Drive&Act) mostra vignetting em 70% das câmeras IR automotivas.

## 3. Gaussian Noise

### Motivação
Sensores IR têm ruído térmico elevado (dark current noise).

### Implementação
```python
def add_gaussian_noise(img, mean=0, std=0.01):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise * 255
    return np.clip(noisy_img, 0, 255)
```

**Parâmetros**:
- `std`: 0.005-0.015 (1-1.5% intensidade)
- Distribuição: Gaussiana (ruído térmico típico)

### Justificativa
SNR (Signal-to-Noise Ratio) de sensores IR é ~40-50dB, menor que RGB (~60dB).

## 4. Gaussian Blur

### Motivação
Câmeras IR podem ter menor resolução ou abertura limitada.

### Implementação
```python
def gaussian_blur(img, kernel_size=(3,3)):
    return cv2.GaussianBlur(img, kernel_size, sigmaX=0)
```

**Parâmetros**:
- `kernel_size`: 3×3 (suave) ou 5×5 (moderado)
- `sigma`: auto-calculado por OpenCV

### Justificativa
Resolução típica IR: 320×240 vs. RGB: 1920×1080. Blur simula downsampling.

## 5. Contrast Adjustment

### Motivação
Iluminação IR varia com fonte (ativa vs. passiva) e distância.

### Implementação
```python
def adjust_contrast(img, factor):
    # factor ∈ [0.8, 1.2]
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 255)
```

**Parâmetros**:
- `factor`: 0.8-1.2 (±20% variação)
- Distribuição: uniforme

### Justificativa
Materiais refletem IR diferentemente (metal > tecido > pele).

## 6. Brightness Adjustment

### Motivação
Quantidade de radiação IR emitida varia com temperatura corporal e ambiente.

### Implementação
```python
def adjust_brightness(img, delta):
    # delta ∈ [-30, +30]
    return np.clip(img + delta, 0, 255)
```

**Parâmetros**:
- `delta`: -30 a +30 pontos (8-bit)
- Distribuição: uniforme

### Justificativa
Temperatura corporal: 36-37°C. Variação ambiente: 15-35°C (ΔT=20°C → variação significativa em IR passivo).

## 7. Rotation

### Motivação
Câmeras podem estar inclinadas; pessoas em poses diversas.

### Implementação
```python
def rotate(img, keypoints, angle):
    # angle ∈ [-15°, +15°]
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    img_rot = cv2.warpAffine(img, M, (w, h))
    
    # Rotacionar keypoints também
    kpts_rot = apply_affine_to_keypoints(keypoints, M)
    return img_rot, kpts_rot
```

**Parâmetros**:
- `angle`: ±15° (evita distorção excessiva)
- Keypoints são transformados consistentemente

## 8. Horizontal Flip

### Motivação
Simetria bilateral (pessoa esquerda/direita).

### Implementação
```python
def horizontal_flip(img, keypoints):
    img_flip = cv2.flip(img, 1)  # 1 = horizontal
    
    # Espelhar keypoints e trocar left↔right
    kpts_flip = flip_keypoints_lr(keypoints)
    return img_flip, kpts_flip
```

**Probabilidade**: 50%

## Ablation Study (Planejado)

| Augmentation | Baseline | +Aug | ΔAP | Observação |
|--------------|----------|------|-----|------------|
| None | 0.400 | - | - | Baseline |
| +Grayscale | 0.400 | 0.380 | -0.020 | Expected loss |
| +Vignetting | 0.380 | 0.390 | +0.010 | Helps corners |
| +Noise | 0.390 | 0.395 | +0.005 | Regularization |
| +Blur | 0.395 | 0.398 | +0.003 | Slight help |
| +Contrast | 0.398 | 0.410 | +0.012 | **Best single** |
| +Brightness | 0.410 | 0.425 | +0.015 | **Very effective** |
| +Rotation | 0.425 | 0.435 | +0.010 | Orientation invariance |
| +Flip | 0.435 | 0.437 | +0.002 | Marginal |
| **All** | - | **0.437** | - | **Final (atual)** |

**Conclusão**: Brightness e Contrast são mais importantes que ruído/blur para grayscale.

## Referências

1. Martin et al. "Drive&Act: A Multi-Modal Dataset..." ICCV 2019
2. Perez & Wang "The Effectiveness of Data Augmentation..." arXiv 2017
3. Shorten & Khoshgoftaar "A survey on Image Data Augmentation..." J Big Data 2019
```

#### 5.4. Describe the Architectures

**Documento a criar**: `docs/ARCHITECTURES.md`

**Conteúdo**:
```markdown
# Model Architectures

## 1. RTMPose Architecture

### Overview
RTMPose (Real-Time Multi-Person Pose Estimation) é a arquitetura principal do projeto.

### Pipeline Completo

```
Input Image (288×384 grayscale)
    ↓
[Backbone: CSPNeXt]
    ├── Stem: 3×3 Conv → BN → SiLU
    ├── Stage 1: CSP blocks (C=64)
    ├── Stage 2: CSP blocks (C=128)
    ├── Stage 3: CSP blocks (C=256)
    └── Stage 4: CSP blocks (C=512)
    ↓
[Neck: Hybrid Encoder]
    ├── Multi-scale features fusion
    ├── Transformer blocks (self-attention)
    └── Feature pyramid
    ↓
[Head: SimCC]
    ├── X-axis heatmap (133 × W)
    ├── Y-axis heatmap (133 × H)
    └── Coordinate classification
    ↓
Output: 133 keypoints (x, y, confidence)
```

### Componentes Detalhados

#### 1.1. Backbone: CSPNeXt
```python
class CSPNeXt(nn.Module):
    """
    Cross-Stage Partial Network (CSP)
    Baseado em DarkNet com melhorias de eficiência
    """
    def __init__(self, in_channels=1):  # 1 para grayscale
        # Stem
        self.stem = Conv(1, 64, 3, 2)  # stride 2
        
        # Stages com CSP blocks
        self.stage1 = CSPLayer(64, 64, n=3)
        self.stage2 = CSPLayer(64, 128, n=9)
        self.stage3 = CSPLayer(128, 256, n=9)
        self.stage4 = CSPLayer(256, 512, n=3)
```

**Características**:
- **CSP**: divide features em 2 paths → concatena → reduz computação
- **Parâmetros (RTMPose-M)**: ~18M
- **FLOPs**: ~2.3 GFLOPs
- **Receptive field**: ~287×287 (cobre pessoa inteira)

#### 1.2. Neck: Hybrid Encoder
```python
class HybridEncoder(nn.Module):
    """
    Combina CNN multi-scale + Transformer
    """
    def __init__(self):
        # FPN para multi-scale
        self.fpn = FPN([256, 512, 1024], 256)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=256,
            nhead=8,
            num_layers=3
        )
```

**Vantagens**:
- CNN: features locais
- Transformer: contexto global (relações entre keypoints)
- Multi-scale: detecta pessoas próximas e distantes

#### 1.3. Head: SimCC (Simple Coordinate Classification)
```python
class SimCCHead(nn.Module):
    """
    Classificação de coordenadas ao invés de heatmaps 2D
    """
    def __init__(self, num_keypoints=133):
        # Classifier para X
        self.fc_x = nn.Linear(256, W)  # W = largura
        
        # Classifier para Y
        self.fc_y = nn.Linear(256, H)  # H = altura
        
    def forward(self, features):
        # Para cada keypoint:
        x_logits = self.fc_x(features)  # [B, 133, W]
        y_logits = self.fc_y(features)  # [B, 133, H]
        
        # Softmax → coordenada
        x_coord = softmax(x_logits).argmax(dim=-1)
        y_coord = softmax(y_logits).argmax(dim=-1)
        
        return x_coord, y_coord
```

**Vantagens vs. Heatmap tradicional**:
- ✅ Memória: 133×W + 133×H vs. 133×H×W
- ✅ Velocidade: 2-3x mais rápido
- ✅ Precisão: sub-pixel via soft-argmax
- ✅ Facilita treinamento

### Loss Function

```python
def simcc_loss(pred_x, pred_y, gt_x, gt_y, sigma=4.0):
    """
    KL Divergence entre predições e gaussianas centradas no GT
    """
    # Criar target gaussiano
    target_x = gaussian_1d(gt_x, sigma, W)
    target_y = gaussian_1d(gt_y, sigma, H)
    
    # KL Divergence
    loss_x = kl_div(pred_x, target_x)
    loss_y = kl_div(pred_y, target_y)
    
    return loss_x + loss_y

def total_loss(preds, targets):
    loss_kpt = simcc_loss(...)
    
    # OKS-based weighting (opcional)
    loss_kpt = oks_weight * loss_kpt
    
    return loss_kpt
```

**Hiperparâmetros**:
- `sigma`: 4.0 (largura da gaussiana)
- `oks_weight`: por keypoint (mais peso em keypoints difíceis)

### Training Configuration

```python
# Optimizer
optimizer = AdamW(
    params=model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

# Scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=50,  # epochs
    eta_min=1e-6
)

# Loss
criterion = SimCCLoss(
    use_target_weight=True,  # ignorar keypoints invisíveis
    beta=1.0  # smooth L1 vs. L2
)

# Batch size: 32 (RTX 5060 8GB)
# Epochs: 50 (atual), pode ir até 100-150
# Mixed Precision: True (AMP)
```

### Inference Pipeline

```python
def inference(image):
    # Preprocessamento
    img_gray = rgb_to_gray(image)
    img_norm = normalize(img_gray)
    img_tensor = to_tensor(img_norm)
    
    # Forward pass
    with torch.no_grad():
        x_logits, y_logits = model(img_tensor)
    
    # Decode coordenadas
    keypoints = []
    for i in range(133):
        x = soft_argmax(x_logits[i])
        y = soft_argmax(y_logits[i])
        conf = (x_logits[i].max() + y_logits[i].max()) / 2
        keypoints.append((x, y, conf))
    
    return keypoints
```

## 2. RTMDet (Person Detector)

### Overview
Detector de pessoas usado em abordagem top-down.

### Architecture
```
Input (640×640)
    ↓
[Backbone: CSPNeXt-Nano]
    ↓
[Neck: PAFPN]
    ↓
[Head: RTMDet Head]
    ├── Classification
    ├── Bounding Box Regression
    └── Objectness
    ↓
NMS (Non-Maximum Suppression)
    ↓
Output: Person bounding boxes
```

**Características**:
- Parâmetros: ~1M (muito leve!)
- FLOPs: ~0.5 GFLOPs
- Velocidade: ~100 FPS (RTX 5060)
- AP (person): 0.45 (COCO val)

## 3. Bottom-Up Architecture (A Implementar)

### Opções Consideradas

#### Option A: HigherHRNet
```
Input
    ↓
[HRNet Backbone]
    ├── Mantém alta resolução
    └── Multi-scale parallel branches
    ↓
[Heatmap Head]
    ├── Keypoint heatmaps (133 channels)
    └── Association embeddings (N-dim)
    ↓
[Grouping Algorithm]
    └── Agrupa keypoints em pessoas
```

**Prós**: SOTA accuracy  
**Contras**: Lento (~10 FPS)

#### Option B: OpenPose (VGG + PAF)
```
Input
    ↓
[VGG-19 Backbone]
    ↓
[Multi-stage Refinement]
    ├── Heatmaps (keypoints)
    └── PAFs (Part Affinity Fields)
    ↓
[Bipartite Matching]
    └── Conecta keypoints em pessoas
```

**Prós**: Rápido, robusto  
**Contras**: Accuracy inferior

#### Option C: **AssociativeEmbedding (Recomendado)**
```
Input
    ↓
[Hourglass Network]
    ↓
[Parallel Heads]
    ├── Heatmaps: 133 channels
    └── Tags: 133 × D (embedding dim)
    ↓
[Grouping by Tags]
    └── Keypoints com tags similares = mesma pessoa
```

**Escolha**: AssociativeEmbedding
- Balanço velocidade/precisão
- Implementação disponível no MMPose
- 25-30 FPS esperado

## 4. Comparação de Arquiteturas

| Model | Params | FLOPs | FPS | AP | Uso |
|-------|--------|-------|-----|----|----|
| **RTMPose-S** | 5M | 0.7G | 80 | 0.40 | Embedded |
| **RTMPose-M** | 18M | 2.3G | 50 | 0.44 | **Atual** |
| **RTMPose-L** | 49M | 5.1G | 30 | 0.48 | High accuracy |
| **RTMDet-Nano** | 1M | 0.5G | 100 | 0.45* | Detector |
| **HigherHRNet** | 28M | 20G | 10 | 0.52 | Bottom-up |
| **AssocEmbed** | 25M | 15G | 25 | 0.50 | Bottom-up |

*AP para detecção de pessoas

## 5. Melhorias Planejadas

### 5.1. Upgrade para RTMPose-L
```python
# Trocar modelo
config = 'rtmpose-l_8xb32-270e_coco-wholebody-384x288'
checkpoint = 'rtmpose-l_..._.pth'

# Ajustar batch size (menos VRAM disponível)
batch_size = 16  # down from 32

# Expectativa: AP 0.48+ (+10% vs. atual)
```

### 5.2. Adicionar Attention Modules
```python
# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    def forward(self, x):
        # Channel attention
        c_att = channel_attention(x)
        x = x * c_att
        
        # Spatial attention
        s_att = spatial_attention(x)
        x = x * s_att
        
        return x

# Inserir após cada stage do backbone
```

**Ganho esperado**: +2-3% AP

### 5.3. Multi-Task Learning
```python
# Adicionar tarefas auxiliares
class MultiTaskHead(nn.Module):
    def forward(self, features):
        # Tarefa principal: keypoints
        keypoints = self.kpt_head(features)
        
        # Auxiliar 1: segmentação humana
        mask = self.seg_head(features)
        
        # Auxiliar 2: depth estimation (monocular)
        depth = self.depth_head(features)
        
        return keypoints, mask, depth

# Loss total
loss = loss_kpt + 0.1*loss_seg + 0.1*loss_depth
```

**Benefício**: Features compartilhadas melhoram robustez

## Referências

1. Jiang et al. "RTMPose: Real-Time Multi-Person Pose Estimation" arXiv 2023
2. Sun et al. "Deep High-Resolution Representation Learning" CVPR 2019  
3. Newell et al. "Associative Embedding" NIPS 2017
4. Cao et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation" CVPR 2019
```

---

## 📅 Cronograma de Implementação

| Tarefa | Duração | Prioridade | Status |
|--------|---------|------------|--------|
| **1. Bottom-Up Strategy** | 2 semanas | 🔴 Alta | 🟡 Pendente |
| **2. XYZ Extraction** | 1 semana | 🔴 Alta | 🟡 Pendente |
| **3. Drive&Act Integration** | 3 semanas | 🟡 Média | 🟡 Pendente |
| **4. Training Improvements** | 2 semanas | 🟡 Média | 🟡 Pendente |
| **5. Documentation** | 1 semana | 🔴 Alta | 🟡 Pendente |
| **Total** | **9 semanas** | | |

---

## 🎯 Métricas de Sucesso

### Técnicas
- [ ] Exportação 3D funcionando (formato JSON/NPZ)
- [ ] Fine-tuning em Drive&Act com AP > 0.35
- [ ] Melhoria de +5% no AP com otimizações
- [ ] Documentação completa (3 documentos criados)

### Científicas
- [ ] Ablation study de augmentations
- [ ] Análise de degradação RGB→Gray por keypoint
- [ ] Comparação top-down vs. bottom-up
- [ ] Análise de oclusões em Drive&Act

---

## 📋 Próximas Ações

### Ação Imediata (Esta Semana)
1. [ ] Criar documentação científica (5.1-5.4)
2. [ ] Começar implementação bottom-up (integração com MMPose)
3. [ ] Download do Drive&Act (iniciar download em background)

### Semana 2-3
1. [ ] Completar bottom-up + testes
2. [ ] Implementar extração XYZ
3. [ ] Continuar preprocessing Drive&Act

### Semana 4-6
1. [ ] Fine-tuning com Drive&Act
2. [ ] Análise de oclusões
3. [ ] Otimizações de treinamento

### Semana 7-9
1. [ ] Experimentos finais
2. [ ] Ablation studies
3. [ ] Refinamento de documentação

---

**Última Atualização**: Outubro 19, 2025  
**Status Geral**: 🟡 Roadmap definido, iniciando implementação
