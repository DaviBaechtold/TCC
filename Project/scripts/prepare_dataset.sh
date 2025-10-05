#!/bin/bash

# Script para preparar o dataset completo
# Executa download, conversão e preparação dos dados

set -e  # Exit on error

echo "=================================="
echo "Dataset Preparation Script"
echo "=================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Diretórios
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"

# Função para print colorido
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Criar diretórios
print_status "Creating directories..."
mkdir -p ${RAW_DIR}
mkdir -p ${PROCESSED_DIR}

# 2. Download COCO-WholeBody
print_status "Step 1: Downloading COCO-WholeBody dataset..."
python src/data/download_coco.py --data-dir ${RAW_DIR}

# Verificar se download foi bem sucedido
if [ $? -ne 0 ]; then
    print_error "Download failed!"
    exit 1
fi

# 3. Aguardar anotações manuais
print_warning "Please download WholeBody annotations manually:"
TRAIN_URL=${WHOLEBODY_TRAIN_URL:-"https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf"}
VAL_URL=${WHOLEBODY_VAL_URL:-"https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgGk0dCTj8xxt"}
echo "  1. Training:   ${TRAIN_URL}"
echo "  2. Validation: ${VAL_URL}"
echo "  Save them to: ${RAW_DIR}/annotations/"
echo ""
read -p "Press Enter when annotations are ready..."

# Verificar anotações
ONLY_TRAIN=0
if [ ! -f "${RAW_DIR}/annotations/coco_wholebody_train_v1.0.json" ]; then
    print_error "Training annotations not found!"
    exit 1
fi

if [ ! -f "${RAW_DIR}/annotations/coco_wholebody_val_v1.0.json" ]; then
    print_warning "Validation annotations not found. Proceeding with TRAIN-only setup (temporary)."
    ONLY_TRAIN=1
else
    print_status "Annotations found!"
fi

# 4. Converter para grayscale
print_status "Step 2: Converting to grayscale (IR simulation)..."
python src/data/convert_to_gray.py \
    --input-dir ${RAW_DIR} \
    --output-dir ${PROCESSED_DIR}/grayscale \
    --method luminosity \
    --simulate-ir \
    --visualize

if [ $? -ne 0 ]; then
    print_error "Conversion failed!"
    exit 1
fi

# 5. Verificar dataset final
print_status "Step 3: Verifying processed dataset..."
python src/data/download_coco.py \
    --data-dir ${PROCESSED_DIR}/grayscale \
    --verify-only

# 6. Estatísticas do dataset
print_status "Step 4: Computing dataset statistics..."

# Contar imagens
NUM_TRAIN=$(find ${PROCESSED_DIR}/grayscale/train2017 -name "*.jpg" | wc -l)
NUM_VAL=$(find ${PROCESSED_DIR}/grayscale/val2017 -name "*.jpg" | wc -l)

echo ""
echo "=================================="
echo "Dataset Statistics"
echo "=================================="
echo "Training images:   ${NUM_TRAIN}"
echo "Validation images: ${NUM_VAL}"
echo "Total images:      $((NUM_TRAIN + NUM_VAL))"
echo ""
echo "Dataset structure:"
echo "${PROCESSED_DIR}/"
echo "├── grayscale/"
echo "│   ├── train2017/    (${NUM_TRAIN} images)"
echo "│   ├── val2017/      (${NUM_VAL} images)"
echo "│   └── annotations/"
echo ""

# 7. Conclusão
print_status "Dataset preparation completed!"
echo ""
echo "Next steps:"
echo "  1. Verify data structure: python src/data/download_coco.py --data-dir data/processed/grayscale --verify-only"
echo "  2. Test augmentations (optional): python src/data/augmentation.py"
if [ "$ONLY_TRAIN" -eq 1 ]; then
    echo "  3. Start training (TEMP train-only): bash -lc \"CONFIG=configs/rtmpose_m_wholebody_train_is_val.py bash scripts/train_full_pipeline.sh\""
    echo "     Note: using train set as validation temporarily; replace with real val JSON when available."
else
    echo "  3. Start training: bash scripts/train_full_pipeline.sh"
fi
echo ""
