# Real-Time 2D Full-Body Pose Estimation for Grayscale (Infrared) Images

## Objetivo
Construir e treinar uma rede para detecção de pose 2D full body em imagens grayscale (infrared) em tempo real, focando em ambientes veiculares.

## Arquitetura
- **Bottom-up approach**: Detecta todos os keypoints primeiro, depois agrupa em pessoas
- **Top-down approach**: Detecta pessoas primeiro (bounding boxes) com RTMDet, depois estima pose com RTMPose

Este projeto utiliza a abordagem **top-down** (RTMDet + RTMPose) para melhor precisão.

## Pipeline
1. **Coleta de dados**: COCO-WholeBody dataset (base principal)
2. **Conversão para Grayscale**: Simular imagens infrared
3. **Data Augmentation**: 
   - Vignetting (simulação de características de câmeras IR)
   - Ruído gaussiano
   - Blur
   - Ajustes de contraste
   - Rotação e flip
4. **Treinamento**: Fine-tuning do RTMPose para entradas grayscale
5. **Avaliação**: Métricas PCK, AP, AR

## Estrutura do Projeto
```
Project/
├── data/
│   ├── raw/              # Dados brutos (COCO-WholeBody)
│   ├── processed/        # Dados processados (grayscale + augmentation)
│   └── annotations/      # Anotações em formato COCO
├── src/
│   ├── data/
│   │   ├── download_coco.py       # Download do dataset
│   │   ├── convert_to_gray.py     # Conversão RGB → Grayscale
│   │   └── augmentation.py        # Data augmentation
│   ├── models/
│   │   ├── rtmdet.py              # Detector de pessoas
│   │   ├── rtmpose.py             # Estimador de pose
│   │   └── pipeline.py            # Pipeline completo
│   ├── training/
│   │   ├── train_detector.py      # Treino do detector
│   │   ├── train_pose.py          # Treino do pose estimator
│   │   └── config.py              # Configurações
│   ├── evaluation/
│   │   ├── metrics.py             # Métricas (PCK, AP, AR)
│   │   └── visualize.py           # Visualização de resultados
│   └── utils/
│       ├── visualization.py       # Funções de visualização
│       └── transforms.py          # Transformações de imagem
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_tests.ipynb
│   └── 03_model_evaluation.ipynb
├── configs/
│   ├── rtmdet_tiny.py             # Config RTMDet tiny
│   └── rtmpose_m.py               # Config RTMPose medium
├── scripts/
│   ├── prepare_dataset.sh         # Script de preparação
│   └── train_full_pipeline.sh     # Script de treinamento
├── requirements.txt
└── README.md
```

## Datasets

### Principal: COCO-WholeBody
- **Descrição**: Extensão do COCO com anotações de corpo completo (133 keypoints)
- **Link**: https://github.com/jin-s13/COCO-WholeBody
- **Keypoints**:
  - Body: 17 keypoints (COCO padrão)
  - Face: 68 keypoints
  - Hands: 42 keypoints (21 cada mão)
  - Feet: 6 keypoints

### Complementares
- **MPII**: Para validação adicional
- **CrowdPose**: Para cenários com múltiplas pessoas
- **3DPW**: Para validação 3D (futuro)

## Conceitos Importantes

### Bottom-Up vs Top-Down

**Bottom-Up**:
- Detecta TODOS os keypoints da imagem primeiro
- Depois agrupa os keypoints em pessoas
- Vantagem: Mais rápido com muitas pessoas
- Desvantagem: Menos preciso
- Exemplo: OpenPose

**Top-Down** (usado neste projeto):
- Detecta pessoas primeiro (bounding boxes)
- Para cada pessoa, estima a pose
- Vantagem: Mais preciso
- Desvantagem: Tempo de processamento cresce com número de pessoas
- Exemplo: RTMDet + RTMPose

### RTMPose Architecture
1. **RTMDet**: Detector de objetos (pessoas) em tempo real
2. **RTMPose**: Estimador de pose de alta precisão
3. **Pipeline**: RTMDet → crop pessoa → RTMPose → keypoints

## Ambiente de Desenvolvimento

### Requisitos
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (para GPU)
- MMPose / MMDetection

### Instalação
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Roadmap

### Fase 1: Setup e Preparação de Dados (Semanas 1-2)
- [ ] Download do COCO-WholeBody
- [ ] Conversão para grayscale
- [ ] Implementação de data augmentation
- [ ] Análise exploratória dos dados

### Fase 2: Baseline Model (Semanas 3-4)
- [ ] Setup RTMDet para detecção de pessoas
- [ ] Setup RTMPose para estimação de pose
- [ ] Treinamento baseline com imagens RGB
- [ ] Avaliação baseline

### Fase 3: Adaptação para Grayscale (Semanas 5-7)
- [ ] Fine-tuning RTMDet para grayscale
- [ ] Fine-tuning RTMPose para grayscale
- [ ] Data augmentation específica para IR
- [ ] Avaliação comparativa RGB vs Grayscale

### Fase 4: Otimização e Tempo Real (Semanas 8-10)
- [ ] Otimização de inferência
- [ ] Quantização do modelo
- [ ] Testes de performance
- [ ] Deployment para tempo real

### Fase 5: Validação e Documentação (Semanas 11-12)
- [ ] Testes em ambiente veicular
- [ ] Documentação completa
- [ ] Artigo científico
- [ ] Apresentação TCC

## Métricas de Avaliação

### Object Keypoint Similarity (OKS)
- Métrica padrão do COCO
- Similar ao IoU, mas para keypoints

### PCK (Percentage of Correct Keypoints)
- Porcentagem de keypoints corretamente detectados
- Threshold geralmente em 0.2 (20% da distância torso)

### AP (Average Precision)
- AP@0.5, AP@0.75, AP@0.5:0.95
- Para diferentes thresholds de OKS

### AR (Average Recall)
- AR@0.5, AR@0.75, AR@0.5:0.95

### Tempo de Inferência
- FPS (Frames Per Second)
- Latência média (ms)
- Throughput

## Referências

### Papers
- RTMPose: Real-Time Multi-Person Pose Estimation (2023)
- COCO-WholeBody: COCO with Whole-Body Keypoint Annotations (2020)
- OpenPose: Realtime Multi-Person 2D Pose Estimation (2018)

### Repositories
- MMPose: https://github.com/open-mmlab/mmpose
- RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody

## Contato
- **Autor**: Davi Baechtold Campos
- **Orientador**: Prof. Dr. Alceu de Souza Brito Junior
- **Instituição**: PUCPR
