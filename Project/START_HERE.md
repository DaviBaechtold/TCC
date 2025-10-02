# ✅ Projeto Criado com Sucesso!

## 📋 O que foi criado

Criei uma estrutura completa para desenvolver um sistema de **detecção de pose 2D full-body para imagens grayscale (infrared) em tempo real**.

### 🗂️ Estrutura do Projeto

```
/home/davs/Documents/TCC/Project/
│
├── 📄 README.md                    # Visão geral do projeto
├── 📄 QUICKSTART.md               # Guia de início rápido detalhado
├── 📄 PROJECT_SUMMARY.md          # Resumo técnico completo
├── 📄 requirements.txt            # Todas as dependências Python
├── 📄 .gitignore                  # Arquivos para ignorar no Git
│
├── 📁 src/                        # Código fonte
│   ├── 📁 data/
│   │   ├── download_coco.py      # Download automático do COCO-WholeBody
│   │   ├── convert_to_gray.py    # Conversão RGB → Grayscale + IR simulation
│   │   └── augmentation.py       # Data augmentation específico para IR
│   ├── 📁 training/
│   │   └── train_pose.py         # Script de treinamento principal
│   ├── 📁 models/                # (a ser criado)
│   ├── 📁 evaluation/            # (a ser criado)
│   └── 📁 utils/                 # (a ser criado)
│
├── 📁 configs/
│   └── rtmpose_m_wholebody.py    # Configuração do modelo RTMPose
│
├── 📁 scripts/
│   └── prepare_dataset.sh        # Script automático de preparação
│
├── 📁 notebooks/                  # (a ser criado)
├── 📁 data/                       # (será criado durante download)
├── 📁 work_dirs/                  # (será criado durante treino)
└── 📁 checkpoints/                # (será criado para modelos pré-treinados)
```

## 🎯 O que o projeto faz

### Objetivo Principal
Treinar uma rede neural (RTMPose) para detectar **133 keypoints de pose humana** em imagens **grayscale/infrared** em tempo real.

### Por que grayscale?
- Simula câmeras infravermelhas (comuns em ambientes veiculares)
- Funciona em baixa luminosidade
- Mais robusto para aplicações de monitoramento interno de veículos

### Abordagem Técnica
1. **Dataset**: COCO-WholeBody (~118K imagens de treino)
2. **Conversão**: RGB → Grayscale com simulação de características IR
3. **Modelo**: RTMPose (top-down approach)
   - RTMDet detecta pessoas
   - RTMPose estima pose de cada pessoa
4. **Augmentation**: Específico para IR (vignetting, ruído térmico, hot pixels)

## 🚀 Próximos Passos (O que VOCÊ deve fazer)

### 1️⃣ Instalar Dependências (15 min)

```bash
# Entrar no diretório do projeto
cd /home/davs/Documents/TCC/Project

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar PyTorch com CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalar outras dependências
pip install -r requirements.txt

# Instalar MMPose e ecosystem
pip install -U openmim
mim install mmengine mmcv mmdet mmpose
```

### 2️⃣ Preparar Dataset (2-4 horas)

```bash
# Dar permissão de execução
chmod +x scripts/prepare_dataset.sh

# Executar script de preparação
bash scripts/prepare_dataset.sh
```

**⚠️ IMPORTANTE**: Durante a execução, o script vai pausar e pedir para você:
1. Baixar manualmente as anotações do Google Drive
2. Salvar em `data/raw/annotations/`
3. Pressionar Enter para continuar

Links das anotações:
- Training: https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf
- Validation: https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgGk0dCTj8xxt

### 3️⃣ Explorar os Dados (30 min)

```bash
# Verificar que tudo foi baixado corretamente
python src/data/download_coco.py --data-dir data/processed/grayscale --verify-only

# Ver estatísticas
ls -lh data/processed/grayscale/train2017/ | wc -l  # Número de imagens treino
ls -lh data/processed/grayscale/val2017/ | wc -l    # Número de imagens val
```

### 4️⃣ Testar Conversão Grayscale (15 min)

```bash
# Converter algumas imagens de teste
python src/data/convert_to_gray.py \
    --input-dir data/raw \
    --output-dir data/test_gray \
    --simulate-ir \
    --visualize
```

Isso vai criar visualizações comparando RGB vs Grayscale.

### 5️⃣ Treinar Modelo Baseline (24-48 horas)

```bash
# Treinar modelo do zero
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --work-dir work_dirs/baseline_grayscale \
    --amp

# OU com fine-tuning (recomendado)
# Primeiro baixar checkpoint pré-treinado
mkdir -p checkpoints
wget -P checkpoints/ https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth

# Depois fine-tune
python src/training/train_pose.py \
    --config configs/rtmpose_m_wholebody.py \
    --load-from checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --work-dir work_dirs/finetune_grayscale \
    --amp
```

### 6️⃣ Monitorar Treinamento

```bash
# Em outro terminal
tensorboard --logdir work_dirs/
```

Abra: http://localhost:6006

## 📚 Documentação

### Ler PRIMEIRO:
1. **PROJECT_SUMMARY.md** - Entender o projeto completo
2. **QUICKSTART.md** - Guia passo-a-passo detalhado

### Conceitos Importantes:

#### Bottom-Up vs Top-Down
- **Bottom-Up**: Detecta todos os keypoints primeiro, depois agrupa em pessoas
  - Exemplo: OpenPose
  - Mais rápido com muitas pessoas
  - Menos preciso

- **Top-Down** (usado aqui):
  - Detecta pessoas primeiro (RTMDet)
  - Depois estima pose de cada pessoa (RTMPose)
  - Mais preciso
  - Ideal para ambientes com poucas pessoas (como veículos)

#### COCO-WholeBody Keypoints
- **Body**: 17 keypoints (COCO padrão)
- **Face**: 68 keypoints
- **Hands**: 42 keypoints (21 por mão)
- **Feet**: 6 keypoints
- **Total**: 133 keypoints

#### Simulação Infrared
As imagens são convertidas para grayscale e recebem:
- **Vignetting**: Escurecimento nas bordas (comum em câmeras IR)
- **Ruído térmico**: Ruído gaussiano
- **Hot pixels**: Pixels defeituosos brilhantes

## 🔧 Troubleshooting Comum

### CUDA Out of Memory
```python
# Editar configs/rtmpose_m_wholebody.py
train_dataloader = dict(
    batch_size=32,  # Reduzir de 64
    ...
)
```

### Dataset não encontrado
```bash
# Verificar estrutura
python src/data/download_coco.py --verify-only
```

### Import errors
```bash
# Reinstalar tudo
pip install -r requirements.txt --force-reinstall
mim install mmengine mmcv mmdet mmpose --force
```

### Expectativas de Resultado

### Hardware do Projeto
- **CPU**: Intel Core i5-14400F (10 cores, 16 threads)
- **GPU**: NVIDIA RTX 5060 8GB (Ada Lovelace architecture) �
- **RAM**: 32GB DDR5 5200MHz
- **OS**: Linux Mint 22.2 Cinnamon (Ubuntu 24.04 base)
- **Storage**: 200GB livres necessários

**Nota**: Seu hardware é **excelente** para este projeto! A RTX 5060 com arquitetura Ada Lovelace é mais eficiente que a RTX 3060, então você pode esperar performance ainda melhor.

### Tempo de Treinamento

### Métricas Esperadas
- **AP (Grayscale)**: 60-65%
- **FPS**: 30-40 (RTX 3060)
- **Latência**: <35ms por frame

## 🎯 Critérios de Sucesso do TCC

1. ✅ AP > 60% em imagens grayscale
2. ✅ FPS ≥ 30 em GPU mid-range
3. ✅ Degradação < 10% vs RGB
4. ✅ Funciona em ambiente veicular

## 📝 Próximas Tarefas (Cronograma)

### Semana 1-2: Setup e Baseline
- [x] Criar estrutura do projeto
- [ ] Instalar dependências
- [ ] Download e preparação de dados
- [ ] Treinar baseline RGB
- [ ] Documentar resultados baseline

### Semana 3-5: Grayscale
- [ ] Conversão completa para grayscale
- [ ] Fine-tuning para grayscale
- [ ] Implementar augmentations IR
- [ ] Comparar RGB vs Grayscale
- [ ] Análise de resultados

### Semana 6-7: Otimização
- [ ] Quantização INT8
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Benchmark tempo real
- [ ] Teste em hardware alvo

### Semana 8-10: Validação
- [ ] Coleta de dados veiculares
- [ ] Teste em ambiente real
- [ ] Análise de casos de falha
- [ ] Fine-tuning com dados reais
- [ ] Documentação final

### Semana 11-12: TCC
- [ ] Escrita do artigo
- [ ] Preparação da apresentação
- [ ] Revisões finais
- [ ] Defesa

## 💡 Dicas Importantes

1. **Comece pequeno**: Teste com subset dos dados primeiro
2. **Monitore sempre**: Use TensorBoard desde o início
3. **Salve checkpoints**: Configure salvamento a cada época
4. **Documente tudo**: Mantenha um log de experimentos
5. **Valide frequentemente**: Rode avaliação a cada 10 epochs

## 🆘 Precisa de Ajuda?

### Recursos:
- **MMPose Docs**: https://mmpose.readthedocs.io/
- **COCO-WholeBody**: https://github.com/jin-s13/COCO-WholeBody
- **RTMPose Paper**: https://arxiv.org/abs/2303.07399

### Issues comuns:
- https://github.com/open-mmlab/mmpose/issues

### Contato:
- Orientador: Prof. Dr. Alceu de Souza Brito Junior

## ✨ Boa Sorte!

Você tem agora uma base sólida para desenvolver seu TCC. O projeto está bem estruturado e segue as melhores práticas da área.

**Próximo passo imediato**: Executar a instalação das dependências!

```bash
cd /home/davs/Documents/TCC/Project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
mim install mmengine mmcv mmdet mmpose
```

---

**Criado por**: GitHub Copilot  
**Data**: Outubro 2025  
**Para**: Davi Baechtold Campos - TCC PUCPR
