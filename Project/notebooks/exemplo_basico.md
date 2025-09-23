# Exemplo Básico - Modelo Multimodal TCC

Este notebook demonstra o uso básico do modelo de fusão multimodal desenvolvido para o TCC.

## Importações

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar componentes do projeto
from src.models.fusion import MultiModalFusionNetwork
from src.data.loaders import VideoDataLoader
from src.utils.visualization import visualize_results
```

## Configuração Básica

```python
# Configurar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")

# Configurações do modelo
config = {
    'depth_model': 'depth_anything_v2',
    'segmentation_model': 'deeplabv3_resnet50',
    'fusion_dim': 512,
    'output_dim': 256,
    'use_temporal': True
}
```

## Inicialização do Modelo

```python
# Criar modelo
model = MultiModalFusionNetwork(
    depth_model=config['depth_model'],
    segmentation_model=config['segmentation_model'],
    fusion_dim=config['fusion_dim'],
    output_dim=config['output_dim'],
    use_temporal=config['use_temporal']
)

model = model.to(device)
print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
```

## Teste com Dados Sintéticos

```python
# Criar dados sintéticos para teste
batch_size = 2
sequence_length = 16
height, width = 224, 224

# Simular sequência de vídeo
video_frames = torch.randn(batch_size, sequence_length, 3, height, width).to(device)
print(f"Frames de vídeo: {video_frames.shape}")

# Simular keypoints (33 pontos do MediaPipe * 3 coordenadas)
keypoints = torch.randn(batch_size, sequence_length, 99).to(device)
print(f"Keypoints: {keypoints.shape}")
```

## Forward Pass

```python
# Teste do modelo
model.eval()
with torch.no_grad():
    # Forward pass básico
    latent_features = model(video_frames, keypoints)
    print(f"Features latentes: {latent_features.shape}")
    
    # Forward pass com features intermediárias
    latent_features, intermediate = model(video_frames, keypoints, return_intermediate=True)
    
    print("\\nFeatures intermediárias:")
    for key, value in intermediate.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
```

## Visualização dos Resultados

```python
# Preparar resultados para visualização
results = {
    'latent_features': latent_features.cpu().numpy(),
    'intermediate_features': {k: v.cpu().numpy() for k, v in intermediate.items() if torch.is_tensor(v)},
    'input_shape': list(video_frames.shape),
    'output_shape': list(latent_features.shape)
}

# Visualizar
visualize_results(results, show_plot=True)
```

## Análise das Features Latentes

```python
# Analisar distribuição das features
features_np = latent_features.cpu().numpy()

plt.figure(figsize=(12, 4))

# Histograma das features
plt.subplot(1, 3, 1)
plt.hist(features_np.flatten(), bins=50, alpha=0.7)
plt.title('Distribuição das Features Latentes')
plt.xlabel('Valor')
plt.ylabel('Frequência')

# Norma das features por amostra
plt.subplot(1, 3, 2)
norms = np.linalg.norm(features_np, axis=1)
plt.bar(range(len(norms)), norms)
plt.title('Norma das Features por Amostra')
plt.xlabel('Amostra')
plt.ylabel('Norma L2')

# Mapa de calor das features
plt.subplot(1, 3, 3)
plt.imshow(features_np, aspect='auto', cmap='viridis')
plt.title('Mapa de Calor das Features')
plt.xlabel('Dimensão da Feature')
plt.ylabel('Amostra')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Teste de Treinamento (Forward + Backward)

```python
# Configurar para treinamento
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Simular um passo de treinamento
optimizer.zero_grad()

# Forward pass
outputs = model(video_frames, keypoints)

# Criar targets dummy (em aplicação real, usar targets reais)
targets = torch.randn_like(outputs)

# Calcular loss
loss = torch.nn.functional.mse_loss(outputs, targets)
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()
optimizer.step()

print("Passo de treinamento concluído!")
```

## Processamento de Vídeo Real (Placeholder)

```python
# Placeholder para processamento de vídeo real
def process_real_video(video_path):
    \"\"\"
    Processa um vídeo real usando o modelo.
    
    Esta é uma implementação placeholder que seria expandida
    para carregar e processar vídeos reais.
    \"\"\"
    print(f"Processando vídeo: {video_path}")
    
    # Em implementação real:
    # 1. Carregar vídeo
    # 2. Extrair frames
    # 3. Pré-processar frames
    # 4. Extrair keypoints com MediaPipe
    # 5. Processar com o modelo
    # 6. Retornar resultados
    
    return {
        'status': 'placeholder',
        'message': 'Implementação completa em desenvolvimento'
    }

# Exemplo de uso
video_path = "path/to/video.mp4"
results = process_real_video(video_path)
print(results)
```

## Próximos Passos

1. **Dados Reais**: Implementar carregamento e pré-processamento de vídeos reais
2. **Treinamento**: Configurar pipeline de treinamento com dados reais
3. **Avaliação**: Implementar métricas de avaliação específicas
4. **Otimização**: Fine-tuning dos hiperparâmetros
5. **Deployment**: Preparar modelo para produção

## Recursos Adicionais

- Configurações em `configs/default.yaml`
- Scripts de treinamento em `scripts/train.py`
- Scripts de avaliação em `scripts/evaluate.py`
- Documentação completa em `docs/`