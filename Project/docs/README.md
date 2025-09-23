# Documentação do Projeto TCC - Geração de Espaço Latente Multimodal

## Visão Geral

Este projeto implementa uma arquitetura de deep learning para geração de espaço latente multimodal, combinando informações de:

- **Estimação de Profundidade Monocular** (Depth Anything 2/Depth Pro)
- **Segmentação Humana** (DeepLabV3/FCN)
- **Pose Estimation** (MediaPipe)
- **Video Embeddings** (3D CNN/Transformer)
- **Análise Temporal** (LSTM/Transformer)

## Arquitetura

### Componentes Principais

1. **Depth Estimator** (`src/models/depth/`)
   - Wrapper para Depth Anything V2
   - Suporte futuro para Depth Pro
   - Pré/pós-processamento automático

2. **Human Segmenter** (`src/models/segmentation/`)
   - Baseado em DeepLabV3 ou FCN
   - Segmentação semântica focada em pessoas
   - Máscaras binárias para isolamento

3. **Pose Estimator** (`src/models/pose/`)
   - Integração com MediaPipe
   - Extração de 33 keypoints (formato MediaPipe)
   - Análise temporal de movimentos
   - Embeddings neurais para poses

4. **Video Embeddings** (`src/models/embeddings/`)
   - Extração de features temporais
   - Suporte para 3D CNN e Transformer
   - Análise de movimento (optical flow)
   - Multi-scale temporal analysis

5. **Fusion Network** (`src/models/fusion/`)
   - Rede principal de fusão multimodal
   - Cross-modal attention mechanisms
   - Geração do espaço latente final

### Pipeline de Processamento

```
Vídeo Input → [Frame Extraction] → [Multimodal Processing] → [Fusion] → Latent Space
                     ↓
            ┌─────────────────────┐
            │  Depth Estimation   │
            │  Human Segmentation │
            │  Pose Estimation    │
            │  Video Embeddings   │
            └─────────────────────┘
                     ↓
            [Cross-Modal Attention]
                     ↓
            [Multimodal Fusion Network]
                     ↓
            [Latent Feature Vector]
```

## Instalação e Configuração

### Dependências

```bash
# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
python scripts/test_basic.py
```

### Configuração

Editar `configs/default.yaml` para ajustar:
- Modelos utilizados
- Hiperparâmetros
- Configurações de dados
- Parâmetros de treinamento

## Uso

### Treinamento

```bash
# Treinamento básico
python scripts/train.py --data_dir /path/to/videos --config configs/default.yaml

# Modo debug (dados sintéticos)
python scripts/train.py --debug

# Continuar treinamento
python scripts/train.py --data_dir /path/to/videos --resume checkpoints/best_model.pth
```

### Avaliação

```bash
# Avaliar modelo
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth --data_dir /path/to/test_videos

# Modo debug
python scripts/evaluate.py --debug --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

### Uso Programático

```python
from src.models.fusion import MultiModalFusionNetwork
from src.data.loaders import VideoDataLoader

# Criar modelo
model = MultiModalFusionNetwork(
    depth_model='depth_anything_v2',
    segmentation_model='deeplabv3_resnet50',
    use_temporal=True
)

# Processar vídeo
results = model.process_video('path/to/video.mp4')
```

## Estrutura de Dados

### Formato de Entrada

- **Vídeos**: MP4, AVI (suportados pelo OpenCV)
- **Keypoints**: JSON/NPZ (formato MediaPipe)
- **Anotações**: JSON com metadados

### Formato de Saída

- **Latent Features**: Tensor (B, output_dim)
- **Intermediate Features**: Dicionário com features de cada modalidade
- **Visualizações**: PNG/HTML

## Experimentação

### Notebooks Disponíveis

- `notebooks/exemplo_basico.md`: Tutorial básico
- Notebooks Jupyter para experimentação interativa (a serem criados)

### Configurações Experimentais

- `configs/default.yaml`: Configuração padrão
- `configs/depth_anything_v2.yaml`: Experimentos com Depth Anything V2

## Desenvolvimento

### Estrutura do Código

```
src/
├── models/              # Modelos de deep learning
│   ├── depth/          # Estimação de profundidade
│   ├── segmentation/   # Segmentação humana
│   ├── pose/           # Pose estimation
│   ├── embeddings/     # Video embeddings
│   └── fusion/         # Fusão multimodal
├── data/               # Data loaders e preprocessing
├── training/           # Pipeline de treinamento
└── utils/             # Utilitários e visualização
```

### Adicionar Novos Modelos

1. Criar módulo em `src/models/[categoria]/`
2. Implementar interface padrão (forward, preprocess, postprocess)
3. Adicionar configuração em `configs/`
4. Atualizar imports em `__init__.py`

### Testes

```bash
# Teste básico
python scripts/test_basic.py

# Testes unitários (quando implementados)
pytest tests/
```

## Monitoramento

### Weights & Biases

- Configurar API key: `wandb login`
- Logs automáticos durante treinamento
- Visualizações interativas
- Comparação de experimentos

### TensorBoard

- Logs locais em `logs/`
- Visualizar: `tensorboard --logdir logs/`

## Troubleshooting

### Problemas Comuns

1. **CUDA Compatibility**: Use CPU se houver problemas com GPU
2. **Memory Issues**: Reduza batch_size e sequence_length
3. **Import Errors**: Verifique PYTHONPATH e instalação de dependências

### Performance

- Use mixed precision training para economia de memória
- Implemente gradient checkpointing para sequências longas
- Considere model parallelism para modelos grandes

## Próximos Passos

1. **Implementação Completa**:
   - HumanSegmenter com modelos reais
   - Integração completa MediaPipe
   - Suporte para Depth Pro

2. **Otimizações**:
   - Model quantization
   - ONNX export
   - TensorRT optimization

3. **Datasets**:
   - Suporte para datasets públicos
   - Ferramentas de anotação
   - Data augmentation específica

4. **Avaliação**:
   - Métricas específicas para cada modalidade
   - Benchmarks comparativos
   - Análise de ablação

## Referências

- Depth Anything V2: [paper/repo]
- MediaPipe: [documentation]
- MMPose/RTMPose: [documentation]
- PyTorch: [documentation]