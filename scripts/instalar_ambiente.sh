#!/usr/bin/env bash
# Cria o ambiente virtual do projeto do zero, na ordem que funciona.
#
# A ordem não é detalhe. Três pacotes antigos do ecossistema OpenMMLab têm
# `setup.py` que importam `pkg_resources`, removido do setuptools recente; o
# MMCV entra como `mmcv-lite`, sem operadores compilados; e o PyTorch precisa vir
# do índice com CUDA 12.8, o único que suporta as GPUs da série RTX 50. Cada
# passo abaixo existe por um desses motivos, e foi verificado numa instalação
# limpa em 24/09/2026.
#
# Uso:  ./scripts/instalar_ambiente.sh [pasta do venv, padrão: venv]
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${1:-venv}"

python3.12 --version >/dev/null 2>&1 \
    || { echo "Python 3.12 não encontrado (sudo apt install python3.12-venv)"; exit 1; }
nvidia-smi >/dev/null 2>&1 \
    || echo "aviso: nvidia-smi não respondeu; sem GPU NVIDIA o sistema não atinge tempo real"

echo "[1/5] ambiente virtual em $VENV"
python3.12 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "[2/5] PyTorch 2.8 com CUDA 12.8"
pip install -q torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[3/5] pacotes com setup.py antigo, sem isolamento de build"
pip install -q "setuptools<81" wheel "cython<3.2" numpy==2.1.2
pip install -q --no-build-isolation mmcv-lite==2.1.0 chumpy==0.70 xtcocotools==1.14.3

echo "[4/5] demais dependências"
pip install -q -r requirements.txt

echo "[5/5] substituto de mmcv._ext"
if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mmcv._ext') else 1)"; then
    echo "  mmcv._ext já existe; nada a fazer"
else
    cp src/models/mmcv_ext_stub.py "$(python -c 'import mmcv, os; print(os.path.dirname(mmcv.__file__))')/_ext.py"
fi

python -c "import torch, mmpose; print(f'  PyTorch {torch.__version__}, CUDA disponível: {torch.cuda.is_available()}, MMPose {mmpose.__version__}')" 2>/dev/null
echo "pronto. Ative com: source $VENV/bin/activate"
