"""
Conversão de imagens RGB para Grayscale (simulação de imagens infrared).

Este script converte as imagens COCO-WholeBody de RGB para grayscale,
simulando as características de câmeras infravermelhas.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import json
import shutil


def rgb_to_gray(image: np.ndarray, method: str = "luminosity") -> np.ndarray:
    """
    Converte imagem RGB para grayscale.
    
    Args:
        image: Imagem RGB (H, W, 3)
        method: Método de conversão
            - "luminosity": Weighted average (0.299R + 0.587G + 0.114B)
            - "average": Simple average
            - "lightness": (max(R,G,B) + min(R,G,B)) / 2
            - "opencv": OpenCV default
            
    Returns:
        Imagem grayscale (H, W)
    """
    if method == "luminosity":
        # Método padrão usado em conversões RGB → Grayscale
        # Pesos baseados em percepção humana de luminosidade
        gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    elif method == "average":
        gray = np.mean(image, axis=2)
    elif method == "lightness":
        gray = (np.max(image, axis=2) + np.min(image, axis=2)) / 2
    elif method == "opencv":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return gray.astype(np.uint8)


def simulate_infrared_characteristics(
    gray: np.ndarray,
    add_noise: bool = True,
    noise_std: float = 5.0,
    vignetting: bool = True,
    vignetting_strength: float = 0.3
) -> np.ndarray:
    """
    Simula características de imagens infravermelhas.
    
    Args:
        gray: Imagem grayscale
        add_noise: Se deve adicionar ruído gaussiano
        noise_std: Desvio padrão do ruído
        vignetting: Se deve aplicar vignetting
        vignetting_strength: Força do vignetting (0-1)
        
    Returns:
        Imagem com características de IR
    """
    result = gray.astype(np.float32)
    
    # Adicionar ruído gaussiano (câmeras IR têm mais ruído)
    if add_noise:
        noise = np.random.normal(0, noise_std, gray.shape)
        result = result + noise
    
    # Aplicar vignetting (escurecimento nas bordas)
    if vignetting:
        h, w = gray.shape
        
        # Criar máscara de vignetting
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        
        # Distância do centro
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        r_norm = r / r_max
        
        # Aplicar vignetting gaussiano
        vignette_mask = 1 - vignetting_strength * (r_norm ** 2)
        result = result * vignette_mask
    
    # Clip para range válido
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def convert_dataset(
    input_dir: str,
    output_dir: str,
    method: str = "luminosity",
    simulate_ir: bool = True,
    copy_annotations: bool = True
):
    """
    Converte dataset completo para grayscale.
    
    Args:
        input_dir: Diretório com imagens RGB
        output_dir: Diretório de saída
        method: Método de conversão
        simulate_ir: Se deve simular características IR
        copy_annotations: Se deve copiar anotações
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Processar train e val
    for split in ["train2017", "val2017"]:
        split_input = input_path / split
        split_output = output_path / split
        
        if not split_input.exists():
            print(f"⚠️  {split} not found, skipping...")
            continue
        
        split_output.mkdir(parents=True, exist_ok=True)
        
        # Listar todas as imagens
        image_files = list(split_input.glob("*.jpg"))
        
        print(f"\n{'=' * 80}")
        print(f"Converting {split}")
        print(f"{'=' * 80}")
        print(f"Input:  {split_input}")
        print(f"Output: {split_output}")
        print(f"Images: {len(image_files)}")
        print(f"Method: {method}")
        print(f"Simulate IR: {simulate_ir}")
        
        # Converter cada imagem
        for img_file in tqdm(image_files, desc=f"Converting {split}"):
            # Ler imagem
            img = cv2.imread(str(img_file))
            
            if img is None:
                print(f"⚠️  Failed to read: {img_file.name}")
                continue
            
            # Converter para grayscale
            gray = rgb_to_gray(img, method=method)
            
            # Simular características IR
            if simulate_ir:
                gray = simulate_infrared_characteristics(
                    gray,
                    add_noise=True,
                    noise_std=3.0,
                    vignetting=True,
                    vignetting_strength=0.2
                )
            
            # Salvar como 3 canais (muitos modelos esperam 3 canais)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            output_file = split_output / img_file.name
            cv2.imwrite(str(output_file), gray_3ch)
        
        print(f"✓ {split} converted!")
    
    # Copiar anotações
    if copy_annotations:
        print(f"\n{'=' * 80}")
        print("Copying annotations")
        print(f"{'=' * 80}")
        
        annotations_input = input_path / "annotations"
        annotations_output = output_path / "annotations"
        
        if annotations_input.exists():
            annotations_output.mkdir(parents=True, exist_ok=True)
            
            for ann_file in annotations_input.glob("*.json"):
                shutil.copy2(ann_file, annotations_output / ann_file.name)
                print(f"✓ Copied: {ann_file.name}")
        else:
            print("⚠️  Annotations not found")
    
    print(f"\n{'=' * 80}")
    print("✅ Conversion completed!")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_path}")


def create_visualization(
    input_dir: str,
    output_dir: str,
    num_samples: int = 10
):
    """
    Cria visualização comparando RGB vs Grayscale.
    
    Args:
        input_dir: Diretório com imagens RGB
        output_dir: Diretório com imagens grayscale
        num_samples: Número de amostras para visualizar
    """
    import matplotlib.pyplot as plt
    
    input_path = Path(input_dir) / "val2017"
    output_path = Path(output_dir) / "val2017"
    
    if not input_path.exists() or not output_path.exists():
        print("⚠️  Directories not found for visualization")
        return
    
    # Selecionar amostras aleatórias
    image_files = list(input_path.glob("*.jpg"))
    samples = np.random.choice(image_files, size=min(num_samples, len(image_files)), replace=False)
    
    # Criar visualização
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    
    for idx, img_file in enumerate(samples):
        # Ler RGB
        rgb = cv2.imread(str(img_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Ler Grayscale
        gray_file = output_path / img_file.name
        gray = cv2.imread(str(gray_file))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"RGB - {img_file.name}")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(gray)
        axes[idx, 1].set_title(f"Grayscale (IR Simulation)")
        axes[idx, 1].axis("off")
    
    plt.tight_layout()
    
    # Salvar visualização
    vis_path = Path(output_dir) / "visualization.png"
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Visualization saved: {vis_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RGB to Grayscale")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Input directory with RGB images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/grayscale",
        help="Output directory for grayscale images"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="luminosity",
        choices=["luminosity", "average", "lightness", "opencv"],
        help="Conversion method"
    )
    parser.add_argument(
        "--simulate-ir",
        action="store_true",
        help="Simulate infrared characteristics"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization"
    )
    
    args = parser.parse_args()
    
    # Converter dataset
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        simulate_ir=args.simulate_ir,
        copy_annotations=True
    )
    
    # Criar visualização
    if args.visualize:
        create_visualization(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_samples=10
        )
