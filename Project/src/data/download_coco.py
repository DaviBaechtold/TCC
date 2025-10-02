"""
Script para download do COCO-WholeBody dataset.

O COCO-WholeBody é uma extensão do COCO com anotações de corpo completo:
- Body: 17 keypoints
- Face: 68 keypoints  
- Hands: 42 keypoints (21 cada)
- Feet: 6 keypoints
Total: 133 keypoints

Referência: https://github.com/jin-s13/COCO-WholeBody
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Barra de progresso para download."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download com barra de progresso."""
    with DownloadProgressBar(
        unit='B', 
        unit_scale=True,
        miniters=1, 
        desc=output_path.split('/')[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, 
            filename=output_path, 
            reporthook=t.update_to
        )


def download_coco_wholebody(data_dir: str = "data/raw"):
    """
    Download do COCO-WholeBody dataset.
    
    Args:
        data_dir: Diretório onde salvar os dados
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # URLs do COCO-WholeBody
    urls = {
        # Imagens COCO 2017
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        
        # Anotações WholeBody
        "annotations_train": "https://drive.google.com/uc?export=download&id=1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf",
        "annotations_val": "https://drive.google.com/uc?export=download&id=1N6VgwKnj8DeyGXCvp1eYgGk0dCTj8xxt",
    }
    
    print("=" * 80)
    print("COCO-WholeBody Dataset Download")
    print("=" * 80)
    
    # Download imagens
    for split, url in [("train2017", urls["train2017"]), 
                       ("val2017", urls["val2017"])]:
        zip_path = data_path / f"{split}.zip"
        images_path = data_path / split
        
        if images_path.exists():
            print(f"\n✓ {split} images already exist. Skipping...")
            continue
            
        print(f"\n📥 Downloading {split} images...")
        download_url(url, str(zip_path))
        
        print(f"📦 Extracting {split}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # Remove zip após extração
        zip_path.unlink()
        print(f"✓ {split} images downloaded and extracted!")
    
    # Download anotações
    annotations_path = data_path / "annotations"
    annotations_path.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Anotações WholeBody")
    print("=" * 80)
    print("\nNOTA: As anotações WholeBody estão no Google Drive.")
    print("Você precisa baixá-las manualmente:")
    print("\n1. Training annotations:")
    print("   https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf")
    print("\n2. Validation annotations:")
    print("   https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgGk0dCTj8xxt")
    print(f"\nSalve os arquivos .json em: {annotations_path}/")
    print("\nEstrutura esperada:")
    print(f"""
{data_dir}/
├── train2017/
│   └── *.jpg
├── val2017/
│   └── *.jpg
└── annotations/
    ├── coco_wholebody_train_v1.0.json
    └── coco_wholebody_val_v1.0.json
    """)
    
    print("\n✅ Download de imagens concluído!")
    print("⚠️  Não esqueça de baixar as anotações manualmente.")


def verify_dataset(data_dir: str = "data/raw"):
    """
    Verifica se o dataset foi baixado corretamente.
    
    Args:
        data_dir: Diretório dos dados
        
    Returns:
        bool: True se dataset está completo
    """
    data_path = Path(data_dir)
    
    required_dirs = [
        data_path / "train2017",
        data_path / "val2017",
        data_path / "annotations"
    ]
    
    required_files = [
        data_path / "annotations" / "coco_wholebody_train_v1.0.json",
        data_path / "annotations" / "coco_wholebody_val_v1.0.json"
    ]
    
    print("\n" + "=" * 80)
    print("Verificando dataset...")
    print("=" * 80)
    
    all_ok = True
    
    # Verificar diretórios
    for dir_path in required_dirs:
        if dir_path.exists():
            if dir_path.name.endswith("2017"):
                num_images = len(list(dir_path.glob("*.jpg")))
                print(f"✓ {dir_path.name}: {num_images} images")
            else:
                print(f"✓ {dir_path.name}: exists")
        else:
            print(f"✗ {dir_path.name}: NOT FOUND")
            all_ok = False
    
    # Verificar arquivos de anotação
    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_path.name}: {size_mb:.1f} MB")
        else:
            print(f"✗ {file_path.name}: NOT FOUND")
            all_ok = False
    
    print("=" * 80)
    if all_ok:
        print("✅ Dataset completo!")
    else:
        print("⚠️  Dataset incompleto. Execute o download novamente.")
    
    return all_ok


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download COCO-WholeBody dataset")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Directory to save dataset"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dataset without downloading"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.data_dir)
    else:
        download_coco_wholebody(args.data_dir)
        verify_dataset(args.data_dir)
