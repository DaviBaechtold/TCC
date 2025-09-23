"""
Wrapper para modelos de estimação de profundidade monocular.
Suporta Depth Anything V2 e Depth Pro.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class DepthAnythingV2(nn.Module):
    """
    Wrapper para o modelo Depth Anything V2.
    Utiliza a implementação do HuggingFace Transformers.
    """
    
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for DepthAnythingV2")
            
        self.model_name = model_name
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def forward(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Estima profundidade para uma imagem.
        
        Args:
            image: Imagem de entrada (H, W, 3) ou (B, 3, H, W)
            
        Returns:
            depth: Mapa de profundidade (H, W) ou (B, 1, H, W)
        """
        # Converter torch tensor para numpy se necessário
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch
                results = []
                for img in image:
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    depth = self.pipe(img_np)["depth"]
                    results.append(torch.from_numpy(np.array(depth)).unsqueeze(0))
                return torch.stack(results)
            else:
                image = image.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
        
        # Estimar profundidade
        result = self.pipe(image)
        depth = np.array(result["depth"])
        
        return torch.from_numpy(depth).unsqueeze(0)


class DepthPro(nn.Module):
    """
    Wrapper para o modelo Depth Pro (placeholder - implementação futura).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        # Placeholder para implementação futura
        print("DepthPro: Implementação em desenvolvimento")
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Placeholder para estimação de profundidade."""
        # Retorna um mapa de profundidade dummy por enquanto
        if image.dim() == 4:
            B, C, H, W = image.shape
            return torch.randn(B, 1, H, W)
        else:
            C, H, W = image.shape
            return torch.randn(1, H, W)


class DepthEstimator(nn.Module):
    """
    Classe unificada para estimação de profundidade.
    Permite alternar entre diferentes modelos.
    """
    
    def __init__(self, model_type: str = "depth_anything_v2", **kwargs):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == "depth_anything_v2":
            self.model = DepthAnythingV2(**kwargs)
        elif model_type == "depth_pro":
            self.model = DepthPro(**kwargs)
        else:
            raise ValueError(f"Modelo não suportado: {model_type}")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Estima profundidade usando o modelo configurado."""
        return self.model(image)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Pré-processa imagem para estimação de profundidade.
        
        Args:
            image: Imagem RGB (H, W, 3)
            
        Returns:
            tensor: Imagem normalizada (3, H, W)
        """
        # Normalizar para [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
            
        # Converter para tensor e reordenar dimensões
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return tensor
    
    def postprocess(self, depth: torch.Tensor) -> np.ndarray:
        """
        Pós-processa mapa de profundidade.
        
        Args:
            depth: Tensor de profundidade
            
        Returns:
            array: Mapa de profundidade normalizado
        """
        depth_np = depth.squeeze().cpu().numpy()
        
        # Normalizar para [0, 1]
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min)
            
        return depth_np