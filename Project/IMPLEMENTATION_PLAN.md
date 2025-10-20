# 📋 Plano de Implementação - Melhorias 2D Pose Estimation

**Data**: Outubro 19, 2025  
**Baseado em**: ROADMAP.md + Discussão com orientador

---

## 🎯 Objetivos Priorizados (Alinhados com Proposta 2D)

### ✅ 1. Real-Time Capture com Multi-Person (70+ FPS)

**Status Atual**:
- ✅ Single-person: ~50 FPS (RTX 5060)
- ✅ Multi-person: ~25 FPS (com RTMDet top-down)
- ❌ Target: 70+ FPS multi-person

**Problema**: Top-down approach é lento para múltiplas pessoas
```
Para N pessoas:
├── 1 detecção (RTMDet): ~10ms
├── N crops: ~2ms × N
├── N pose estimations: ~20ms × N
└── Total: 10 + 22N ms

Para 5 pessoas: ~120ms = 8 FPS ❌
```

**Solução**: Implementar estratégia híbrida

#### Abordagem 1: Bottom-Up com AssociativeEmbedding
```python
# Pipeline otimizado
Input (640×480)
    ↓
[Backbone: RTMPose-S] (~15ms)
    ↓
[Parallel Heads]
    ├── Heatmaps: 133 keypoints (~5ms)
    └── Embeddings: agrupamento (~3ms)
    ↓
[Grouping Algorithm] (~3ms)
    └── Keypoints → Pessoas
    ↓
Output: N pessoas, 133 kpts cada
    
Total: ~26ms = 38 FPS (para qualquer N!)
```

**Limitação**: Precisão ~5% menor que top-down

#### Abordagem 2: Otimizações Top-Down
```python
# Mantém precisão, melhora velocidade

1. TensorRT Optimization
   ├── RTMDet: 10ms → 3ms
   └── RTMPose: 20ms → 8ms
   
2. Batch Processing
   ├── Processar N pessoas em 1 batch
   └── 20ms × N → 25ms total
   
3. GPU Async Streams
   ├── Overlap detection + pose
   └── ~30% speedup

Total: ~15ms = 66 FPS (até 3-4 pessoas)
```

#### Abordagem 3: Híbrido Adaptativo (RECOMENDADO)
```python
def adaptive_inference(frame, persons_detected):
    if persons_detected <= 2:
        # Top-down (mais preciso)
        return rtmdet_rtmpose_pipeline(frame)
    else:
        # Bottom-up (mais rápido)
        return associative_embedding(frame)

# Melhor dos dois mundos:
# - 1-2 pessoas: precisão máxima (~50 FPS)
# - 3+ pessoas: velocidade (~35 FPS)
```

**Implementação Prioritária**:

**Fase 1.1: TensorRT Export (1 semana)**
```bash
# Converter modelos para TensorRT
cd /home/davs/Documents/TCC/Project

# RTMDet
python -m mmdeploy.tools.deploy \
    configs/detection/tensorrt_dynamic.py \
    configs/detectors/rtmdet_nano_person_infer.py \
    checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    data/raw/val2017/000000000139.jpg \
    --work-dir deploy/rtmdet_trt \
    --device cuda:0

# RTMPose
python -m mmdeploy.tools.deploy \
    configs/pose/tensorrt_dynamic.py \
    configs/rtmpose_m_wholebody_minimal.py \
    work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
    data/raw/val2017/000000000139.jpg \
    --work-dir deploy/rtmpose_trt \
    --device cuda:0
```

**Fase 1.2: Batch Processing (3 dias)**
```python
# src/evaluation/run_realtime_optimized.py

def batch_inference(bboxes, frame):
    """
    Processar múltiplas pessoas em 1 batch
    """
    # Crop todas as pessoas
    crops = [crop_bbox(frame, bbox) for bbox in bboxes]
    
    # Resize para tamanho uniforme
    crops_resized = [cv2.resize(c, (288, 384)) for c in crops]
    
    # Stack em batch
    batch = torch.stack([to_tensor(c) for c in crops_resized])
    
    # Inferência batch (1 forward pass!)
    with torch.no_grad():
        keypoints_batch = model(batch)
    
    # Desnormalizar para coordenadas originais
    keypoints_list = []
    for i, bbox in enumerate(bboxes):
        kpts = denormalize(keypoints_batch[i], bbox)
        keypoints_list.append(kpts)
    
    return keypoints_list
```

**Fase 1.3: Async GPU Streams (2 dias)**
```python
import torch.cuda as cuda

# Criar streams CUDA
stream_det = cuda.Stream()
stream_pose = cuda.Stream()

def async_inference(frame):
    # Stream 1: Detecção
    with cuda.stream(stream_det):
        bboxes = detector(frame)
    
    # Stream 2: Pose (overlap com próxima detecção)
    with cuda.stream(stream_pose):
        keypoints = pose_estimator(frame, bboxes)
    
    # Sincronizar apenas quando necessário
    cuda.synchronize()
    
    return keypoints
```

**Métricas de Sucesso**:
- [ ] 70+ FPS para 1-2 pessoas
- [ ] 35+ FPS para 3-5 pessoas
- [ ] Degradação de AP < 5%
- [ ] Latência < 30ms (1-2 pessoas)

---

### ✅ 2. Extração para Plano Cartesiano XYZ

**Objetivo**: Projetar keypoints 2D em coordenadas 3D normalizadas.

**Abordagem**: Weak Perspective Projection + Heurísticas Anatômicas

#### Fase 2.1: Normalização 2D → 3D (3 dias)

```python
# src/utils/projection_3d.py

import numpy as np
from scipy.spatial.transform import Rotation

class WeakPerspective3DProjector:
    """
    Projeta keypoints 2D em 3D usando weak perspective + priors anatômicos
    """
    
    def __init__(self):
        # Prior anatômico: distâncias relativas típicas
        self.skeleton_priors = {
            'torso_length': 0.45,      # 45cm (normalizado)
            'arm_length': 0.60,        # 60cm
            'leg_length': 0.90,        # 90cm
            'shoulder_width': 0.40,    # 40cm
            'hip_width': 0.30          # 30cm
        }
        
        # Keypoint visibility → profundidade relativa
        # Ex: braços à frente do tronco, pés atrás
        self.depth_priors = {
            'nose': 0.1,               # frente
            'shoulders': 0.0,          # referência
            'elbows': 0.05,            # ligeiramente à frente
            'wrists': 0.10,            # à frente
            'hips': -0.05,             # ligeiramente atrás
            'knees': -0.10,            # atrás
            'ankles': -0.15            # mais atrás
        }
    
    def project_to_3d(self, keypoints_2d, bbox):
        """
        keypoints_2d: [133, 3] (x, y, confidence)
        bbox: [x1, y1, x2, y2]
        
        Returns:
        keypoints_3d: [133, 4] (X, Y, Z, confidence)
        """
        # 1. Extrair root joint (hip center)
        left_hip = keypoints_2d[11]   # COCO idx
        right_hip = keypoints_2d[12]
        root = (left_hip + right_hip) / 2
        
        # 2. Normalizar por root (translação)
        kpts_centered = keypoints_2d[:, :2] - root[:2]
        
        # 3. Normalizar por escala (bbox diagonal)
        bbox_size = np.linalg.norm([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        kpts_normalized = kpts_centered / bbox_size
        
        # 4. Estimar Z (profundidade) por região
        z_coords = self._estimate_depth(kpts_normalized, keypoints_2d)
        
        # 5. Montar coordenadas 3D
        keypoints_3d = np.zeros((133, 4))
        keypoints_3d[:, 0] = kpts_normalized[:, 0]  # X
        keypoints_3d[:, 1] = kpts_normalized[:, 1]  # Y
        keypoints_3d[:, 2] = z_coords                # Z
        keypoints_3d[:, 3] = keypoints_2d[:, 2]      # confidence
        
        return keypoints_3d
    
    def _estimate_depth(self, kpts_2d, kpts_orig):
        """
        Estima coordenada Z usando heurísticas
        """
        z = np.zeros(133)
        
        # Body keypoints (0-16)
        body_indices = list(range(17))
        for idx in body_indices:
            joint_name = self._get_joint_name(idx)
            z[idx] = self.depth_priors.get(joint_name, 0.0)
        
        # Face keypoints (17-90): ligeiramente à frente
        z[23:91] = 0.12  # face à frente
        
        # Hand keypoints (91-132)
        # Inferir profundidade por distância do ombro
        left_hand = list(range(91, 112))
        right_hand = list(range(112, 133))
        
        left_shoulder = kpts_2d[5]
        right_shoulder = kpts_2d[6]
        
        for idx in left_hand:
            dist = np.linalg.norm(kpts_2d[idx] - left_shoulder)
            z[idx] = 0.05 + 0.3 * dist  # mais longe → mais à frente
        
        for idx in right_hand:
            dist = np.linalg.norm(kpts_2d[idx] - right_shoulder)
            z[idx] = 0.05 + 0.3 * dist
        
        # Suavizar Z por consistência temporal (se vídeo)
        # z = self._temporal_smoothing(z)  # TODO
        
        return z
    
    def _get_joint_name(self, idx):
        """Mapear índice COCO para nome anatômico"""
        coco_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        name = coco_names[idx] if idx < 17 else 'other'
        
        # Mapear para categoria de profundidade
        if 'shoulder' in name: return 'shoulders'
        if 'elbow' in name: return 'elbows'
        if 'wrist' in name: return 'wrists'
        if 'hip' in name: return 'hips'
        if 'knee' in name: return 'knees'
        if 'ankle' in name: return 'ankles'
        return 'nose'
    
    def export_to_json(self, keypoints_3d, output_path):
        """Salvar em formato JSON"""
        data = {
            'version': '1.0',
            'keypoints_3d': keypoints_3d.tolist(),
            'coordinate_system': {
                'origin': 'hip_center',
                'unit': 'normalized (bbox diagonal = 1.0)',
                'axes': 'X: right, Y: down, Z: forward'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_to_npz(self, keypoints_3d, output_path):
        """Salvar em formato NumPy"""
        np.savez_compressed(
            output_path,
            keypoints_3d=keypoints_3d,
            format='XYZ_confidence'
        )
    
    def visualize_3d(self, keypoints_3d):
        """Visualização 3D com matplotlib"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot keypoints
        X = keypoints_3d[:, 0]
        Y = keypoints_3d[:, 1]
        Z = keypoints_3d[:, 2]
        conf = keypoints_3d[:, 3]
        
        # Colorir por confiança
        scatter = ax.scatter(X, Y, Z, c=conf, cmap='viridis', s=50)
        
        # Desenhar skeleton
        # (adicionar linhas entre keypoints conectados)
        
        ax.set_xlabel('X (right)')
        ax.set_ylabel('Y (down)')
        ax.set_zlabel('Z (forward)')
        ax.set_title('3D Pose Projection')
        
        plt.colorbar(scatter, label='Confidence')
        plt.show()
```

#### Fase 2.2: Integração com Pipeline Realtime (2 dias)

```python
# src/evaluation/run_realtime_3d.py

from src.utils.projection_3d import WeakPerspective3DProjector

def main():
    # Inicializar
    projector = WeakPerspective3DProjector()
    
    # Loop de captura
    while True:
        ret, frame = cap.read()
        
        # Detecção + Pose 2D
        bboxes = detector(frame)
        keypoints_2d = pose_estimator(frame, bboxes)
        
        # Projeção 3D
        keypoints_3d_list = []
        for i, (bbox, kpts_2d) in enumerate(zip(bboxes, keypoints_2d)):
            kpts_3d = projector.project_to_3d(kpts_2d, bbox)
            keypoints_3d_list.append(kpts_3d)
            
            # Salvar frame
            if args.save_3d:
                projector.export_to_json(
                    kpts_3d,
                    f'output/frame_{frame_id}_person_{i}.json'
                )
        
        # Visualização 2D (como antes)
        draw_keypoints(frame, keypoints_2d)
        
        # Visualização 3D (opcional, janela separada)
        if args.show_3d:
            projector.visualize_3d(keypoints_3d_list[0])
        
        cv2.imshow('2D Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

**Limitações (documentar claramente)**:
- ⚠️ **Não é lifting real**: sem treino 2D→3D, apenas heurísticas
- ⚠️ **Profundidade relativa**: Z não é métrico (sem escala absoluta)
- ⚠️ **Ambiguidade**: poses simétricas podem ter Z invertido
- ✅ **Útil para**: visualização, análise qualitativa, features para classificação

**Métricas de Sucesso**:
- [ ] Exportação JSON/NPZ funcionando
- [ ] Visualização 3D sem erros
- [ ] Teste em 100 frames: Z consistente temporalmente
- [ ] Documentação clara das limitações

---

### ✅ 3. Integração Drive&Act Dataset

**Prioridade**: ALTA (validação em cenário veicular real)

#### Fase 3.1: Download e Preprocessing (1 semana)

```bash
# scripts/download_driveact.sh

#!/bin/bash

# Drive&Act dataset
# Fonte: https://www.driveact.com/

echo "🚗 Downloading Drive&Act Dataset..."

# Criar estrutura
mkdir -p data/driveact/{raw,processed}

# Download (requer registro)
# NOTA: Dataset é ~200GB, pode levar horas
wget --user=YOUR_USER --password=YOUR_PASS \
    https://www.driveact.com/downloads/driveact_full.tar.gz \
    -O data/driveact/raw/driveact_full.tar.gz

# Extrair subset relevante (apenas câmeras side/dash)
tar -xzf data/driveact/raw/driveact_full.tar.gz \
    --wildcards '*/camera_side/*' '*/camera_dash/*' \
    -C data/driveact/raw/

echo "✅ Download completo!"

# Preprocessing
python src/data/preprocess_driveact.py \
    --input-dir data/driveact/raw \
    --output-dir data/driveact/processed \
    --views side dash \
    --extract-frames \
    --fps 5  # 5 FPS (suficiente para pose estimation)
```

```python
# src/data/preprocess_driveact.py

import json
import cv2
from pathlib import Path
from tqdm import tqdm

class DriveActPreprocessor:
    """
    Converte Drive&Act para formato COCO-like
    """
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Mapeamento de anotações Drive&Act → COCO
        self.action_to_pose_map = {
            'reaching_side': 'arm_extended',
            'drinking': 'hand_to_mouth',
            'phone_call': 'hand_to_ear',
            # ... 83 classes totais
        }
    
    def extract_frames(self, video_path, output_dir, fps=5):
        """Extrair frames de vídeo"""
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frame_id = 0
        saved = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % frame_interval == 0:
                # Converter para grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Salvar
                output_path = output_dir / f'frame_{saved:06d}.jpg'
                cv2.imwrite(str(output_path), gray)
                saved += 1
            
            frame_id += 1
        
        cap.release()
        return saved
    
    def convert_annotations(self):
        """
        Converter anotações Drive&Act para COCO format
        """
        # Drive&Act tem anotações de ações, não poses diretas
        # Estratégia: usar ações como proxy para poses esperadas
        
        coco_format = {
            'info': {
                'description': 'Drive&Act converted to COCO-WholeBody format',
                'version': '1.0'
            },
            'images': [],
            'annotations': [],
            'categories': self._get_coco_categories()
        }
        
        # Ler anotações originais
        annotation_file = self.input_dir / 'annotations.json'
        with open(annotation_file) as f:
            driveact_annos = json.load(f)
        
        # Converter cada frame
        for img_id, anno in enumerate(tqdm(driveact_annos)):
            # Imagem
            coco_format['images'].append({
                'id': img_id,
                'file_name': anno['frame_path'],
                'width': anno['width'],
                'height': anno['height']
            })
            
            # Anotação (se temos pose GT)
            if 'keypoints' in anno:
                coco_format['annotations'].append({
                    'id': img_id,
                    'image_id': img_id,
                    'category_id': 1,  # person
                    'keypoints': anno['keypoints'],  # 133 keypoints
                    'num_keypoints': anno['num_visible'],
                    'bbox': anno['bbox'],
                    'area': anno['area']
                })
        
        # Salvar
        output_file = self.output_dir / 'annotations_coco.json'
        with open(output_file, 'w') as f:
            json.dump(coco_format, f)
        
        print(f"✅ Conversão completa: {len(coco_format['images'])} imagens")
    
    def _get_coco_categories(self):
        """Categorias COCO padrão"""
        return [{
            'id': 1,
            'name': 'person',
            'keypoints': [
                'nose', 'left_eye', 'right_eye', ...  # 133 total
            ],
            'skeleton': [
                [16, 14], [14, 12], ...  # conexões
            ]
        }]
```

#### Fase 3.2: Fine-tuning em Drive&Act (1 semana)

```python
# configs/rtmpose_m_driveact.py

_base_ = ['./rtmpose_m_wholebody_minimal.py']

# Dataset Drive&Act
data_root = 'data/driveact/processed/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations_coco_train.json',
        data_prefix=dict(img='images/'),
    ))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations_coco_val.json',
        data_prefix=dict(img='images/'),
    ))

# Load do modelo COCO
load_from = 'work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth'

# Fine-tuning com learning rate menor
optim_wrapper = dict(
    optimizer=dict(lr=1e-4)  # 10x menor que treino inicial
)

# Menos epochs (domain adaptation)
train_cfg = dict(max_epochs=20)
```

```bash
# Treinar
python src/training/train_pose.py \
    --config configs/rtmpose_m_driveact.py \
    --work-dir work_dirs/driveact_finetune \
    --amp
```

#### Fase 3.3: Análise de Oclusões (3 dias)

```python
# src/evaluation/analyze_occlusions.py

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class OcclusionAnalyzer:
    """
    Analisa impacto de oclusões veiculares em keypoints
    """
    
    def __init__(self):
        self.occlusion_regions = {
            'steering_wheel': [9, 10, 11, 12],  # wrists, hips
            'dashboard': [13, 14, 15, 16],       # knees, ankles
            'seat': [11, 12, 13, 14]             # hips, knees
        }
    
    def analyze_dataset(self, predictions, ground_truth):
        """
        Compara predições vs. GT para identificar padrões de erro
        """
        errors_by_keypoint = defaultdict(list)
        errors_by_region = defaultdict(list)
        
        for pred, gt in zip(predictions, ground_truth):
            # Erro por keypoint
            for i in range(133):
                if gt[i, 2] > 0:  # visível no GT
                    error = np.linalg.norm(pred[i, :2] - gt[i, :2])
                    errors_by_keypoint[i].append(error)
            
            # Erro por região de oclusão
            for region, indices in self.occlusion_regions.items():
                region_error = []
                for idx in indices:
                    if gt[idx, 2] > 0:
                        error = np.linalg.norm(pred[idx, :2] - gt[idx, :2])
                        region_error.append(error)
                if region_error:
                    errors_by_region[region].append(np.mean(region_error))
        
        # Visualizar
        self.plot_errors(errors_by_keypoint, errors_by_region)
    
    def plot_errors(self, errors_kpt, errors_region):
        """Plotar análise de erros"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Erro por keypoint
        kpt_indices = sorted(errors_kpt.keys())
        mean_errors = [np.mean(errors_kpt[i]) for i in kpt_indices]
        
        ax1.bar(kpt_indices, mean_errors)
        ax1.set_xlabel('Keypoint Index')
        ax1.set_ylabel('Mean Error (pixels)')
        ax1.set_title('Error by Keypoint')
        ax1.axhline(y=np.mean(mean_errors), color='r', linestyle='--', 
                    label='Overall Mean')
        ax1.legend()
        
        # Erro por região de oclusão
        regions = list(errors_region.keys())
        region_means = [np.mean(errors_region[r]) for r in regions]
        
        ax2.bar(regions, region_means)
        ax2.set_ylabel('Mean Error (pixels)')
        ax2.set_title('Error by Occlusion Region')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/driveact_occlusion_analysis.png', dpi=300)
        plt.show()

# Uso
analyzer = OcclusionAnalyzer()
analyzer.analyze_dataset(predictions, ground_truth)
```

**Métricas de Sucesso**:
- [ ] AP > 0.35 em Drive&Act val set
- [ ] Documentar degradação COCO→Drive&Act (esperado: 10-15%)
- [ ] Identificar top-3 keypoints mais afetados por oclusões
- [ ] Propor melhorias específicas (ex: attention em regiões ocluídas)

---

### ✅ 4. Melhorar Treinamento da Rede Neural

**Objetivo**: AP 0.50+ (atual: 0.4373)

#### Estratégia 4.1: Upgrade Arquitetura (1 semana)

```bash
# Testar RTMPose-L (maior capacidade)
python src/training/train_pose.py \
    --config configs/rtmpose_l_wholebody.py \
    --load-from checkpoints/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth \
    --work-dir work_dirs/rtmpose_l_grayscale \
    --amp
```

#### Estratégia 4.2: Data Augmentation Avançada (3 dias)

```python
# src/data/augmentation_advanced.py

class AdvancedAugmentation:
    """
    Técnicas state-of-the-art de augmentation
    """
    
    def __init__(self):
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
    
    def mixup(self, img1, img2, kpts1, kpts2):
        """
        Mixup: mistura linear de duas imagens
        Referência: Zhang et al. "mixup: Beyond Empirical Risk Minimization"
        """
        lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Misturar imagens
        img_mixed = lambda_ * img1 + (1 - lambda_) * img2
        
        # Misturar keypoints (soft labels)
        kpts_mixed = lambda_ * kpts1 + (1 - lambda_) * kpts2
        
        return img_mixed, kpts_mixed
    
    def cutmix(self, img1, img2, kpts1, kpts2):
        """
        CutMix: recortar região de uma imagem e colar em outra
        Referência: Yun et al. "CutMix: Regularization Strategy..."
        """
        h, w = img1.shape[:2]
        
        # Região aleatória
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_w = int(w * np.sqrt(lam))
        cut_h = int(h * np.sqrt(lam))
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Misturar
        img_mixed = img1.copy()
        img_mixed[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        
        # Keypoints: manter apenas se não estão na região recortada
        kpts_mixed = kpts1.copy()
        for i in range(len(kpts1)):
            x, y = kpts1[i][:2]
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Keypoint na região recortada → usar kpts2
                kpts_mixed[i] = kpts2[i]
        
        return img_mixed, kpts_mixed
    
    def random_erasing(self, img, kpts):
        """
        Random Erasing: simula oclusões
        Referência: Zhong et al. "Random Erasing Data Augmentation"
        """
        h, w = img.shape[:2]
        
        # Parâmetros
        area = h * w
        target_area = np.random.uniform(0.02, 0.2) * area
        aspect_ratio = np.random.uniform(0.3, 3.0)
        
        # Dimensões do patch
        patch_h = int(np.sqrt(target_area * aspect_ratio))
        patch_w = int(np.sqrt(target_area / aspect_ratio))
        
        if patch_h < h and patch_w < w:
            x = np.random.randint(0, w - patch_w)
            y = np.random.randint(0, h - patch_h)
            
            # Apagar região
            img[y:y+patch_h, x:x+patch_w] = np.random.randint(0, 255)
            
            # Marcar keypoints ocluídos
            for i in range(len(kpts)):
                kx, ky = kpts[i][:2]
                if x <= kx <= x+patch_w and y <= ky <= y+patch_h:
                    kpts[i, 2] = 0  # marcar como invisível
        
        return img, kpts
```

#### Estratégia 4.3: Loss Functions Avançadas (2 dias)

```python
# src/training/losses.py

import torch
import torch.nn as nn

class WingLoss(nn.Module):
    """
    Wing Loss: melhor para keypoints difíceis (hands, face)
    Referência: Feng et al. "Wing Loss for Robust Facial Landmark Localisation"
    """
    
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
    
    def forward(self, pred, target):
        delta = (pred - target).abs()
        
        # Wing loss é suave perto de zero, linear longe
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        return loss.mean()

class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss: pesos adaptativos por keypoint
    """
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target, keypoint_weights):
        """
        keypoint_weights: [133] tensor com pesos por keypoint
        Ex: mãos/face têm peso maior
        """
        delta = (pred - target).abs()
        
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-target))) * \
            (self.alpha-target) * ((self.theta/self.epsilon)**(self.alpha-target-1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1+self.theta/self.epsilon)
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + (delta/self.epsilon)**(self.alpha-target)),
            A * delta - C
        )
        
        # Aplicar pesos por keypoint
        weighted_loss = loss * keypoint_weights.view(1, -1, 1)
        
        return weighted_loss.mean()

# Configuração de pesos por região
keypoint_weights = torch.ones(133)
keypoint_weights[23:91] *= 1.5   # Face (mais difícil)
keypoint_weights[91:133] *= 1.5  # Hands (mais difícil)
keypoint_weights[0:17] *= 1.0    # Body (mais fácil)
```

#### Estratégia 4.4: Training Tricks (1 semana)

```python
# configs/rtmpose_m_advanced.py

# 1. EMA (Exponential Moving Average)
custom_hooks = [
    dict(
        type='EMAHook',
        momentum=0.0002,
        priority='ABOVE_NORMAL'
    )
]

# 2. Learning Rate Warmup + Cosine Annealing
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500  # 500 iterations warmup
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=150,
        eta_min=1e-6,
        begin=0,
        end=150,
        by_epoch=True
    )
]

# 3. Label Smoothing
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=150,
    val_interval=5,
    val_begin=10  # começar validação após 10 epochs
)

# 4. Gradient Clipping
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2)  # clip gradients
)

# 5. More Epochs
train_cfg = dict(max_epochs=150)  # vs. 50 atual

# 6. Early Stopping
early_stopping = dict(
    monitor='coco-wholebody/AP',
    patience=20,
    mode='max'
)
```

**Meta Realista**:
- Atual: AP 0.4373
- Com L: AP ~0.48 (+10%)
- Com augmentation: AP ~0.50 (+15%)
- Com loss/tricks: AP ~0.52 (+19%)

**Métricas de Sucesso**:
- [ ] AP > 0.50 (target principal)
- [ ] AP_hands > 0.40 (atualmente mais baixo)
- [ ] AP_face > 0.55 (melhorar detecção facial)
- [ ] Treinamento estável (loss convergindo suavemente)

---

### ✅ 5. Documentação Científica Completa

#### Fase 5.1: Review of 2D Keypoint Metrics (2 dias)

**Criar**: `docs/METRICS.md` (já esboçado no ROADMAP)

**Conteúdo**:
- OKS (Object Keypoint Similarity) - fórmula + interpretação
- PCK (Percentage Correct Keypoints) - múltiplos thresholds
- MPJPE (Mean Per Joint Position Error) - em pixels
- Análise por região (body, face, hands, feet)
- Curvas ROC/PR

#### Fase 5.2: Describe the Datasets (2 dias)

**Criar**: `docs/DATASETS.md` (já esboçado no ROADMAP)

**Conteúdo**:
- COCO-WholeBody: estatísticas completas
- Drive&Act: especificações técnicas
- Preprocessamento aplicado
- Train/val/test splits
- Desafios e limitações

#### Fase 5.3: Describe Data Augmentations (2 dias)

**Criar**: `docs/AUGMENTATIONS.md` (já esboçado no ROADMAP)

**Conteúdo**:
- 8 técnicas implementadas
- Justificativa científica de cada uma
- Parâmetros e probabilidades
- Ablation study (qual contribui mais?)
- Comparação com/sem augmentation

#### Fase 5.4: Describe Architectures (2 dias)

**Criar**: `docs/ARCHITECTURES.md` (já esboçado no ROADMAP)

**Conteúdo**:
- RTMPose-M: diagrama + detalhes
- CSPNeXt backbone
- Hybrid Encoder
- SimCC head
- Comparação RTMPose-S/M/L
- Bottom-up alternatives

---

## 📅 Cronograma Consolidado

| Semana | Tarefa Principal | Sub-tarefas | Entregáveis |
|--------|------------------|-------------|-------------|
| **1** | Real-time Optimization | TensorRT export, Batch processing | 70+ FPS multi-person |
| **2** | 3D Projection | WeakPerspective class, Exportação JSON/NPZ | XYZ extraction working |
| **3** | Drive&Act Download | Download, Extract frames | Preprocessed dataset |
| **4** | Drive&Act Preprocessing | Convert annotations, Split dataset | COCO-format ready |
| **5** | Drive&Act Fine-tuning | Train, Evaluate | Model checkpoint |
| **6** | Occlusion Analysis | Analyze errors, Visualize | Analysis report |
| **7** | Training Improvements | RTMPose-L, Advanced aug | AP > 0.50 |
| **8** | Training Refinement | Loss functions, Training tricks | Best model |
| **9** | Documentation | Write all 4 docs | Scientific documentation |
| **10** | Integration & Testing | End-to-end testing, Bug fixes | Complete system |

**Total**: 10 semanas (~2.5 meses)

---

## 🎯 Deliverables Finais

### Técnicos
- [ ] Real-time system: 70+ FPS multi-person
- [ ] 3D extraction: JSON/NPZ export working
- [ ] Drive&Act model: AP > 0.35
- [ ] Best model: AP > 0.50 on COCO
- [ ] 4 documentation files created

### Científicos
- [ ] Ablation study: augmentations
- [ ] Comparative analysis: COCO vs. Drive&Act
- [ ] Occlusion analysis: vehicular scenario
- [ ] Architecture comparison: RTMPose-M vs. L vs. bottom-up

### Demonstração
- [ ] Video demo: real-time webcam (70+ FPS)
- [ ] Video demo: Drive&Act scenes
- [ ] 3D visualization: rotating poses
- [ ] Slides: resultados + gráficos

---

## ⚠️ Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Drive&Act download falha | Média | Alto | Usar subset ou dataset alternativo |
| TensorRT não atinge 70 FPS | Baixa | Médio | Fallback: PyTorch JIT + otimizações |
| AP não chega a 0.50 | Média | Médio | Documentar tentativas, aceitar 0.48+ |
| Drive&Act sem GT de pose | Alta | Alto | Re-anotar subset OU usar apenas para inferência |
| Tempo insuficiente | Baixa | Alto | Priorizar itens 1, 2, 5 (core) |

---

## 📞 Próximas Ações (Esta Semana)

### Segunda-feira
- [ ] Commit do ROADMAP e IMPLEMENTATION_PLAN
- [ ] Começar TensorRT export (Item 1.1)
- [ ] Registrar no site Drive&Act para download

### Terça-feira
- [ ] Implementar batch processing (Item 1.2)
- [ ] Testar FPS multi-person
- [ ] Iniciar download Drive&Act (background)

### Quarta-feira
- [ ] Implementar WeakPerspective3DProjector (Item 2.1)
- [ ] Testes de projeção 3D

### Quinta-feira
- [ ] Integrar 3D no pipeline realtime
- [ ] Exportação JSON/NPZ funcionando

### Sexta-feira
- [ ] Começar documentação (METRICS.md)
- [ ] Revisar progresso da semana
- [ ] Ajustar cronograma se necessário

---

**Criado em**: Outubro 19, 2025  
**Status**: 🟢 Plano aprovado, iniciando implementação  
**Próxima Revisão**: Outubro 26, 2025 (fim da Semana 1)
