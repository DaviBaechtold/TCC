# TCC - Geração de Espaço Latente Multimodal

Projeto de TCC focado na geração de espaço latente combinando estimação de profundidade monocular, segmentação humana, processamento multi-view e análise temporal com video embeddings.

## Visão Geral

Este projeto implementa uma arquitetura modular inspirada no MMPose/RTMPose que integra:

- **Estimação de Profundidade Monocular**: Depth Anything 2 ou Depth Pro
- **Segmentação Humana**: Segmentação semântica para isolamento de pessoas
- **Processamento Multi-view**: Análise de múltiplas perspectivas
- **Video Embeddings**: Extração de features temporais para análise de vídeo
- **Keypoints MediaPipe**: Integração de pontos-chave para pose estimation

## Estrutura do Projeto

```
Project/
├── src/
│   ├── models/
│   │   ├── depth/           # Modelos de estimação de profundidade
│   │   ├── segmentation/    # Modelos de segmentação humana
│   │   ├── pose/           # Modelos de pose estimation
│   │   ├── fusion/         # Redes de fusão multimodal
│   │   └── embeddings/     # Video embeddings e feature extraction
│   ├── data/
│   │   ├── loaders/        # Data loaders e preprocessamento
│   │   ├── transforms/     # Transformações de dados
│   │   └── datasets/       # Classes de dataset customizadas
│   ├── utils/
│   │   ├── visualization/  # Utilitários de visualização
│   │   ├── metrics/        # Métricas de avaliação
│   │   └── io/            # Utilitários de I/O
│   └── training/
│       ├── trainers/       # Classes de treinamento
│       ├── schedulers/     # Learning rate schedulers
│       └── losses/         # Funções de loss customizadas
├── configs/                # Arquivos de configuração
├── scripts/               # Scripts de treinamento e avaliação
├── notebooks/            # Jupyter notebooks para experimentação
├── tests/               # Testes unitários
└── docs/               # Documentação do projeto
```

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Dependências Principais

- **PyTorch**: Framework principal de deep learning
- **OpenCV**: Processamento de imagem e vídeo
- **MediaPipe**: Extração de keypoints e pose estimation
- **Transformers**: Para modelos de depth estimation
- **Albumentations**: Augmentações de dados
- **Weights & Biases**: Logging e monitoramento de experimentos
- **NumPy, SciPy**: Computação científica
- **Matplotlib, Seaborn**: Visualização

## Uso Rápido

```python
from src.models.fusion import MultiModalFusionNetwork
from src.data.loaders import VideoDataLoader
from src.utils.visualization import visualize_results

# Carregar dados
data_loader = VideoDataLoader(config_path='configs/default.yaml')

# Criar modelo de fusão
model = MultiModalFusionNetwork(
    depth_model='depth_anything_v2',
    segmentation_model='deeplabv3',
    pose_model='mediapipe'
)

# Processar vídeo
results = model.process_video(video_path='path/to/video.mp4')

# Visualizar resultados
visualize_results(results)
```

## Configuração

Os experimentos são configurados através de arquivos YAML em `configs/`. Exemplo:

```yaml
model:
  depth:
    name: "depth_anything_v2"
    pretrained: true
  segmentation:
    name: "deeplabv3_resnet50"
    num_classes: 21
  fusion:
    hidden_dim: 512
    num_layers: 3

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 100
  device: "cuda"
```

## Contribuição

Este é um projeto de TCC. Para questões ou sugestões, entre em contato através do repositório.

## Licença

Este projeto é desenvolvido para fins acadêmicos como parte de um Trabalho de Conclusão de Curso.