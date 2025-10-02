"""
Data Augmentation para simulação de características de imagens infrared.

Implementa augmentations específicas para:
- Simular características de câmeras IR
- Vignetting
- Ruído
- Variações de contraste
- Blur
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional, List
import random


class VignettingTransform(A.ImageOnlyTransform):
    """
    Aplica efeito de vignetting (escurecimento nas bordas).
    Comum em câmeras infravermelhas.
    """
    
    def __init__(
        self,
        strength: float = 0.3,
        always_apply: bool = False,
        p: float = 0.5
    ):
        """
        Args:
            strength: Força do vignetting (0-1)
            always_apply: Se deve sempre aplicar
            p: Probabilidade de aplicação
        """
        super().__init__(always_apply, p)
        self.strength = strength
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        
        # Criar máscara de vignetting
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        
        # Distância do centro
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        r_norm = r / r_max
        
        # Aplicar vignetting gaussiano
        vignette_mask = 1 - self.strength * (r_norm ** 2)
        vignette_mask = np.expand_dims(vignette_mask, axis=-1)
        
        # Aplicar máscara
        result = (img * vignette_mask).astype(np.uint8)
        
        return result


class ThermalNoiseTransform(A.ImageOnlyTransform):
    """
    Adiciona ruído térmico típico de câmeras infravermelhas.
    """
    
    def __init__(
        self,
        noise_std: float = 10.0,
        always_apply: bool = False,
        p: float = 0.5
    ):
        """
        Args:
            noise_std: Desvio padrão do ruído
            always_apply: Se deve sempre aplicar
            p: Probabilidade de aplicação
        """
        super().__init__(always_apply, p)
        self.noise_std = noise_std
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Gerar ruído gaussiano
        noise = np.random.normal(0, self.noise_std, img.shape)
        
        # Adicionar ruído
        result = img.astype(np.float32) + noise
        
        # Clip para range válido
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


class HotPixelTransform(A.ImageOnlyTransform):
    """
    Simula hot pixels (pixels defeituosos comuns em câmeras IR).
    """
    
    def __init__(
        self,
        num_pixels: int = 10,
        always_apply: bool = False,
        p: float = 0.3
    ):
        """
        Args:
            num_pixels: Número de hot pixels a adicionar
            always_apply: Se deve sempre aplicar
            p: Probabilidade de aplicação
        """
        super().__init__(always_apply, p)
        self.num_pixels = num_pixels
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()
        
        # Adicionar hot pixels aleatórios
        for _ in range(self.num_pixels):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            value = random.randint(200, 255)  # Pixels brilhantes
            
            result[y, x] = value
        
        return result


def get_training_augmentation(
    image_size: tuple = (256, 192),
    infrared_simulation: bool = True
) -> A.Compose:
    """
    Pipeline de augmentation para treinamento.
    
    Args:
        image_size: Tamanho da imagem (height, width)
        infrared_simulation: Se deve incluir simulações IR
        
    Returns:
        Compose de transformações
    """
    transforms = []
    
    # Resize e crop
    transforms.extend([
        A.LongestMaxSize(max_size=max(image_size)),
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.RandomCrop(height=image_size[0], width=image_size[1]),
    ])
    
    # Augmentations geométricas
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ])
    
    # Augmentations de cor/intensidade
    transforms.extend([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.8),
    ])
    
    # Simulações específicas de infrared
    if infrared_simulation:
        transforms.extend([
            VignettingTransform(strength=0.3, p=0.5),
            ThermalNoiseTransform(noise_std=8.0, p=0.6),
            HotPixelTransform(num_pixels=5, p=0.3),
        ])
    
    # Blur e ruído
    transforms.extend([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    ])
    
    # Normalização
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    )


def get_validation_augmentation(
    image_size: tuple = (256, 192)
) -> A.Compose:
    """
    Pipeline de augmentation para validação.
    
    Args:
        image_size: Tamanho da imagem (height, width)
        
    Returns:
        Compose de transformações
    """
    return A.Compose([
        A.LongestMaxSize(max_size=max(image_size)),
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.CenterCrop(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ))


def get_test_time_augmentation() -> List[A.Compose]:
    """
    Pipeline de TTA (Test Time Augmentation).
    
    Returns:
        Lista de composições de transformações
    """
    return [
        # Original
        A.Compose([A.NoOp()]),
        
        # Flip horizontal
        A.Compose([A.HorizontalFlip(p=1.0)]),
        
        # Multi-scale
        A.Compose([A.RandomScale(scale_limit=0.1, p=1.0)]),
        A.Compose([A.RandomScale(scale_limit=-0.1, p=1.0)]),
    ]


def visualize_augmentations(
    image: np.ndarray,
    keypoints: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualiza exemplos de augmentations.
    
    Args:
        image: Imagem original (H, W, 3)
        keypoints: Keypoints (N, 2)
        num_samples: Número de amostras a gerar
        save_path: Caminho para salvar visualização
    """
    import matplotlib.pyplot as plt
    
    # Get augmentation pipeline
    transform = get_training_augmentation(infrared_simulation=True)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    for idx in range(num_samples):
        # Apply augmentation
        transformed = transform(
            image=image,
            keypoints=keypoints
        )
        
        aug_image = transformed['image']
        aug_keypoints = np.array(transformed['keypoints'])
        
        # Denormalize image
        aug_image = aug_image.permute(1, 2, 0).numpy()
        aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_image = np.clip(aug_image, 0, 1)
        
        # Plot original
        if idx == 0:
            axes[0, idx].imshow(image)
            axes[0, idx].scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20)
            axes[0, idx].set_title("Original")
            axes[0, idx].axis('off')
        
        # Plot augmented
        axes[1, idx].imshow(aug_image)
        if len(aug_keypoints) > 0:
            axes[1, idx].scatter(aug_keypoints[:, 0], aug_keypoints[:, 1], c='red', s=20)
        axes[1, idx].set_title(f"Augmented {idx + 1}")
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test augmentations
    print("Testing data augmentation pipeline...")
    
    # Create dummy image and keypoints
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    keypoints = np.random.rand(17, 2) * np.array([640, 480])
    
    # Get transforms
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()
    
    print("✓ Training augmentation pipeline created")
    print("✓ Validation augmentation pipeline created")
    
    # Apply transforms
    train_result = train_transform(image=image, keypoints=keypoints)
    val_result = val_transform(image=image, keypoints=keypoints)
    
    print(f"✓ Train transform output shape: {train_result['image'].shape}")
    print(f"✓ Val transform output shape: {val_result['image'].shape}")
    print(f"✓ Train keypoints: {len(train_result['keypoints'])} points")
    print(f"✓ Val keypoints: {len(val_result['keypoints'])} points")
    
    print("\n✅ All tests passed!")
