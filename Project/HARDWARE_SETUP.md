# 🖥️ Configuração de Hardware e Software

## 💻 Especificações do Sistema

### Hardware
```
CPU:     Intel Core i5-14400F
         - 10 cores (6P + 4E)
         - 16 threads
         - Base: 2.5 GHz, Boost: 4.7 GHz
         - Cache: 20MB L3
         - TDP: 65W (base)

GPU:     NVIDIA GeForce RTX 5060
         - 8GB GDDR6 VRAM
         - Arquitetura: Ada Lovelace
         - CUDA Cores: ~4352
         - Tensor Cores: 136 (Gen 4)
         - RT Cores: 34 (Gen 3)
         - Memory Bandwidth: ~288 GB/s
         - TDP: ~115W

RAM:     32GB DDR5 5200MHz
         - Dual Channel
         - Latência: ~CL40-42

Storage: SSD recomendado
         - Mínimo 200GB livres
         - NVMe recomendado para I/O rápido
```

### Sistema Operacional
```
OS:      Linux Mint 22.2 Cinnamon
Base:    Ubuntu 24.04 LTS (Noble Numbat)
Kernel:  6.8+ (default)
DE:      Cinnamon 6.2+
```

---

## 🚀 Performance Esperada

### Comparação com Baseline (RTX 3060)

| Métrica | RTX 3060 | RTX 5060 | Melhoria |
|---------|----------|----------|----------|
| **FPS (single)** | 30-35 | 35-50 | +15-40% |
| **Latência** | 30-35ms | 25-30ms | ~15% |
| **Batch (32)** | 35 img/s | 40-50 img/s | +15-40% |
| **Treinamento** | 48h | 36-42h | ~20% |
| **VRAM** | 8GB | 8GB | Igual |
| **Tensor Cores** | Gen 3 | Gen 4 | +30% |

### Otimizações Específicas RTX 5060

A RTX 5060 com arquitetura **Ada Lovelace** traz melhorias:

1. **Tensor Cores Gen 4**
   - FP16: ~2x mais rápido que Gen 3
   - INT8: ~3x mais rápido
   - Melhor para Mixed Precision Training

2. **DLSS Frame Generation** (não usado diretamente, mas indica eficiência)
   - Melhor para inferência em tempo real

3. **Eficiência Energética**
   - ~20% mais eficiente por watt
   - Menos throttling térmico

4. **PCIe 4.0**
   - Menor latência de transferência CPU↔GPU

---

## 🔧 Instalação do Ambiente

### 1. Verificar Drivers NVIDIA (Linux Mint 22.2)

```bash
# Verificar GPU
lspci | grep -i nvidia

# Verificar driver instalado
nvidia-smi

# Se não instalado, instalar driver mais recente
sudo apt update
sudo apt install nvidia-driver-560  # ou versão mais recente disponível
sudo reboot

# Após reboot, verificar
nvidia-smi
```

**Output esperado**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.xx                 Driver Version: 560.xx         CUDA Version: 12.6    |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060      Off |   00000000:01:00.0  On |                  N/A |
|  0%   35C    P8              8W /  115W |    500MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### 2. Instalar CUDA Toolkit 12.6

```bash
# Adicionar repositório NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Instalar CUDA
sudo apt install cuda-toolkit-12-6

# Adicionar ao PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar instalação
nvcc --version
```

### 3. Instalar cuDNN 9.x

```bash
# Download do site NVIDIA (requer conta)
# https://developer.nvidia.com/cudnn

# Ou via apt (se disponível)
sudo apt install cudnn9-cuda-12
```

### 4. Instalar Python 3.11

```bash
# Linux Mint 22.2 já vem com Python 3.11+
python3 --version  # Deve mostrar 3.11.x ou 3.12.x

# Instalar pip e venv
sudo apt install python3-pip python3-venv python3-dev

# Instalar dependências de build
sudo apt install build-essential git cmake
```

### 5. Criar Ambiente Virtual

```bash
cd /home/davs/Documents/TCC/Project

# Criar venv
python3 -m venv venv

# Ativar
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip setuptools wheel
```

### 6. Instalar PyTorch para CUDA 12.6

```bash
# PyTorch com CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verificar instalação
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Output esperado**:
```
PyTorch: 2.5.0+cu126
CUDA available: True
CUDA version: 12.6
GPU: NVIDIA GeForce RTX 5060
```

### 7. Instalar Dependências do Projeto

```bash
# Instalar requirements
pip install -r requirements.txt

# Instalar MMPose ecosystem
pip install -U openmim
mim install mmengine mmcv mmdet mmpose

# Verificar instalações
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
python -c "import mmdet; print(f'MMDet: {mmdet.__version__}')"
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"
```

---

## 📊 Benchmark Inicial

### Teste de GPU

```bash
# Criar script de teste
cat > test_gpu.py << 'EOF'
import torch
import time

def benchmark_gpu():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Teste de throughput
    batch_size = 64
    img_size = (3, 256, 192)
    
    # Warm up
    x = torch.randn(batch_size, *img_size, device=device)
    for _ in range(10):
        y = torch.nn.functional.conv2d(x, torch.randn(64, 3, 3, 3, device=device))
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        y = torch.nn.functional.conv2d(x, torch.randn(64, 3, 3, 3, device=device))
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = 100 * batch_size / elapsed
    print(f"\nBenchmark Results:")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {fps:.1f} images/second")
    print(f"Latency: {1000 * elapsed / 100:.2f}ms per batch")
    
    # Verificar VRAM
    print(f"\nVRAM Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"VRAM Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == '__main__':
    benchmark_gpu()
EOF

python test_gpu.py
```

**Output esperado (RTX 5060)**:
```
GPU: NVIDIA GeForce RTX 5060
VRAM Total: 8.00 GB

Benchmark Results:
Time: 2.35s
Throughput: 2723.4 images/second
Latency: 23.50ms per batch

VRAM Usage: 0.85 GB
VRAM Cached: 1.20 GB
```

---

## ⚡ Otimizações Específicas

### 1. Mixed Precision (FP16)

**Recomendado**: Sempre usar `--amp` flag no treinamento

```bash
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/baseline \
    --amp  # ← Ativa FP16
```

**Benefícios**:
- ~2x mais rápido
- ~50% menos VRAM
- Mesma precisão final

### 2. Batch Size Otimizado

Com 8GB VRAM, você pode usar:

```python
# configs/rtmpose_m_wholebody.py
train_dataloader = dict(
    batch_size=80,      # ← Aumentar de 64 para 80
    num_workers=10,     # ← i5-14400F tem 16 threads
    ...
)
```

### 3. DataLoader Otimizado

```python
# Aproveitar os 10 cores do i5-14400F
train_dataloader = dict(
    batch_size=80,
    num_workers=10,           # ← 10 workers para 10 cores
    persistent_workers=True,  # ← Manter workers ativos
    pin_memory=True,          # ← Acelera transferência CPU→GPU
    prefetch_factor=2,        # ← Pre-fetch 2 batches
    ...
)
```

### 4. Compilação JIT (PyTorch 2.0+)

```python
# No início do train_pose.py
import torch
model = torch.compile(model, mode='reduce-overhead')
```

**Benefício**: ~10-15% mais rápido

---

## 🔥 Configuração Otimizada Final

### configs/rtmpose_m_wholebody_optimized.py

```python
# Otimizado para i5-14400F + RTX 5060 + 32GB RAM

# Data loaders otimizados
train_dataloader = dict(
    batch_size=80,              # Maior batch com RTX 5060
    num_workers=10,             # 10 workers para 10 cores
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    ...
)

val_dataloader = dict(
    batch_size=64,              # Batch maior para validação
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    ...
)

# Mixed precision
optim_wrapper = dict(
    type='AmpOptimWrapper',     # Ativa FP16 automaticamente
    loss_scale='dynamic',
    ...
)

# Gradient accumulation (se necessário)
train_cfg = dict(
    accumulation_steps=1,       # Sem acumulação com batch 80
    ...
)
```

---

## 📈 Monitoramento de Performance

### Durante Treinamento

```bash
# Terminal 1: Treinar
python src/training/train_pose.py --config configs/rtmpose_m_wholebody.py --amp

# Terminal 2: Monitorar GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitorar recursos
htop
```

### Métricas a Observar

```
GPU Utilization:  > 95% ✅ (ideal)
GPU Memory:       4-6 GB (de 8GB)
GPU Temp:         < 80°C (thermal throttle em ~83°C)
CPU Usage:        40-60% (DataLoaders)
RAM Usage:        8-12 GB (de 32GB)
```

---

## 🎯 Performance Esperada (Resumo)

### Treinamento
- **Baseline RGB**: 36-40 horas (420 epochs)
- **Fine-tune Gray**: 18-22 horas (200 epochs)
- **FPS durante treino**: ~1.2-1.5 epoch/hour

### Inferência
- **Single image**: 25-30ms
- **Batch 32**: ~40-50 img/s
- **Real-time (webcam)**: 35-50 FPS

### VRAM Usage
- **Treinamento (batch 80)**: ~5-6 GB
- **Inferência (batch 1)**: ~1-2 GB
- **Margem disponível**: 2-3 GB

---

## ✅ Checklist de Verificação

```bash
# 1. Driver NVIDIA
nvidia-smi

# 2. CUDA
nvcc --version

# 3. Python
python3 --version

# 4. PyTorch + CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 5. GPU detectada
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 6. MMPose
python -c "import mmpose; print(mmpose.__version__)"

# 7. Dataset
ls -lh data/processed/grayscale/train2017/ | wc -l

# Tudo OK? ✅ Pronto para treinar!
```

---

## 🆘 Troubleshooting Específico

### Driver NVIDIA não carrega
```bash
# Remover drivers antigos
sudo apt purge nvidia-* -y
sudo apt autoremove -y

# Reinstalar
sudo apt install nvidia-driver-560
sudo reboot
```

### CUDA out of memory
```bash
# Reduzir batch size
# Em configs/rtmpose_m_wholebody.py
batch_size=64  # de 80 para 64
```

### CPU bottleneck (baixo uso de GPU)
```bash
# Aumentar num_workers
num_workers=12  # de 10 para 12
```

### Thermal throttling
```bash
# Monitorar temperatura
watch -n 1 nvidia-smi

# Se > 80°C:
# 1. Melhorar cooling
# 2. Reduzir power limit
sudo nvidia-smi -pl 100  # Limita a 100W (padrão 115W)
```

---

## 🎉 Setup Completo!

Com este hardware você tem um **excelente** setup para o projeto:
- ✅ GPU moderna e eficiente (Ada Lovelace)
- ✅ RAM abundante (32GB)
- ✅ CPU com bom número de cores (10C/16T)
- ✅ SO Linux otimizado para deep learning

**Performance esperada**: Melhor que a maioria dos papers de referência! 🚀

---

**Última atualização**: Outubro 2025  
**Hardware**: i5-14400F + RTX 5060 + 32GB DDR5 + Linux Mint 22.2
