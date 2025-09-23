"""
TCC - Geração de Espaço Latente Multimodal

Este pacote implementa uma arquitetura modular para geração de espaço latente
combinando estimação de profundidade monocular, segmentação humana, 
processamento multi-view e análise temporal.
"""

__version__ = "0.1.0"
__author__ = "Davi Bächtold"
__email__ = "davi.baechtold@example.com"

# Imports condicionais para evitar erros quando dependências não estão instaladas
try:
    from .models.fusion import MultiModalFusionNetwork
    from .data.loaders import VideoDataLoader
    from .utils.visualization import visualize_results
    
    __all__ = [
        "MultiModalFusionNetwork",
        "VideoDataLoader", 
        "visualize_results"
    ]
except ImportError as e:
    print(f"Aviso: Algumas dependências não estão instaladas: {e}")
    __all__ = []