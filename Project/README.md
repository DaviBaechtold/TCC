# Real-Time 2D Full-Body Pose Estimation for Grayscale (Infrared) Images

## Objetivo
Construir e treinar uma rede para detecção de pose 2D full body em imagens grayscale (infrared) em tempo real, focando em ambientes veiculares.

## Arquitetura
- **Bottom-up approach**: Detecta todos os keypoints primeiro, depois agrupa em pessoas
- **Top-down approach**: Detecta pessoas primeiro (bounding boxes) com RTMDet, depois estima pose com RTMPose

Este projeto utiliza a abordagem **top-down** (RTMDet + RTMPose) para melhor precisão.

## Pipeline
1. **Coleta de dados**: COCO-WholeBody dataset (base principal)
2. **Conversão para Grayscale**: Simular imagens infrared
3. **Data Augmentation**: 
   - Vignetting (simulação de características de câmeras IR)
   - Ruído gaussiano
   - Blur
   - Ajustes de contraste
   - Rotação e flip
4. **Treinamento**: Fine-tuning do RTMPose para entradas grayscale
5. **Avaliação**: Métricas PCK, AP, AR

## Estrutura do Projeto
```
Project/
├── data/
│   ├── raw/              # Dados brutos (COCO-WholeBody)
│   ├── processed/        # Dados processados (grayscale + augmentation)
│   └── annotations/      # Anotações em formato COCO
├── src/
│   ├── data/
│   │   ├── download_coco.py       # Download do dataset
│   │   ├── convert_to_gray.py     # Conversão RGB → Grayscale
│   │   └── augmentation.py        # Data augmentation
│   ├── models/
│   │   ├── rtmdet.py              # Detector de pessoas
│   │   ├── rtmpose.py             # Estimador de pose
│   │   └── pipeline.py            # Pipeline completo
│   ├── training/
│   │   ├── train_detector.py      # Treino do detector
│   │   ├── train_pose.py          # Treino do pose estimator
│   │   └── config.py              # Configurações
│   ├── evaluation/
│   │   ├── metrics.py             # Métricas (PCK, AP, AR)
│   │   └── visualize.py           # Visualização de resultados
│   └── utils/
│       ├── visualization.py       # Funções de visualização
│       └── transforms.py          # Transformações de imagem
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_tests.ipynb
│   └── 03_model_evaluation.ipynb
├── configs/
│   ├── rtmdet_tiny.py             # Config RTMDet tiny
│   └── rtmpose_m.py               # Config RTMPose medium
├── scripts/
│   ├── prepare_dataset.sh         # Script de preparação
│   └── train_full_pipeline.sh     # Script de treinamento
├── requirements.txt
└── README.md
```

## Datasets

### Principal: COCO-WholeBody
- **Descrição**: Extensão do COCO com anotações de corpo completo (133 keypoints)
- **Link**: https://github.com/jin-s13/COCO-WholeBody
- **Keypoints**:
  - Body: 17 keypoints (COCO padrão)
  - Face: 68 keypoints
  - Hands: 42 keypoints (21 cada mão)
  - Feet: 6 keypoints

### Complementares
- **MPII**: Para validação adicional
- **CrowdPose**: Para cenários com múltiplas pessoas
- **3DPW**: Para validação 3D (futuro)

## Conceitos Importantes

### Bottom-Up vs Top-Down

**Bottom-Up**:
- Detecta TODOS os keypoints da imagem primeiro
- Depois agrupa os keypoints em pessoas
- Vantagem: Mais rápido com muitas pessoas
- Desvantagem: Menos preciso
- Exemplo: OpenPose

**Top-Down** (usado neste projeto):
- Detecta pessoas primeiro (bounding boxes)
- Para cada pessoa, estima a pose
- Vantagem: Mais preciso
- Desvantagem: Tempo de processamento cresce com número de pessoas
- Exemplo: RTMDet + RTMPose

### RTMPose Architecture
1. **RTMDet**: Detector de objetos (pessoas) em tempo real
2. **RTMPose**: Estimador de pose de alta precisão
3. **Pipeline**: RTMDet → crop pessoa → RTMPose → keypoints

## Ambiente de Desenvolvimento

### Requisitos
- Python 3.8+ (recomendado 3.10-3.12)
- GPU com drivers NVIDIA compatíveis (recomendado CUDA 12.x para as instruções abaixo)
- Espaço livre em disco: >= 100GB (datasets + checkpoints)

### Instalação (passos testados)
As instruções abaixo foram testadas neste repositório e funcionaram em um ambiente Linux com CUDA 12.6.

```bash
# Entre no diretório do projeto
cd /home/davs/Documents/TCC/Project

# 1) Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2) Atualizar pip
pip install --upgrade pip

# 3) Instalar PyTorch + CUDA (exemplo para CUDA 12.6 / PyTorch 2.8)
# Ajuste a URL conforme sua CUDA/PyTorch alvo: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 4) Instalar openmim (helper do OpenMMLab) e dependências principais
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
mim install mmpose

# 5) Instalar bibliotecas do requirements (opcional/auxiliar)
pip install -r requirements.txt

# 6) Verificar instalação básica
python -c "import torch, mmcv, mmdet, mmpose, mmengine; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmpose.__version__, mmengine.__version__)"
```

Observação: trocas de versão entre `mmcv`, `mmdet`, `mmengine` e `mmpose` são sensíveis — use as faixas recomendadas acima. Se você preferir uma alternativa isolada, usar uma imagem Docker oficial do OpenMMLab (quando disponível) evita problemas de compilação local.

## Execução rápida (exemplos)

Ative o venv (`source venv/bin/activate`) e use o Python do ambiente. Exemplos:

1) Testar importações (sanity check):

```bash
python -c "import torch, mmcv, mmdet, mmpose, mmengine; print('OK')"
```

2) Treinar / fine-tune (com checkpoint pré-baixado):

```bash
# Coloque o checkpoint em checkpoints/
python src/training/train_pose.py \
  --config configs/rtmpose_m_wholebody.py \
  --load-from checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
  --work-dir work_dirs/finetune_grayscale \
  --amp
```

3) Rodar um dry-run (teste rápido) com a configuração mínima que acompanha o projeto:

```bash
python src/training/train_pose.py \
  --config configs/rtmpose_m_wholebody_minimal.py \
  --work-dir work_dirs/test_minimal
```

Se quiser forçar o uso da GPU verifique que o venv Python reconhece a GPU (veja `python -c "import torch; print(torch.cuda.is_available())"`) e rode os comandos com o venv ativado; o script seleciona GPUs automaticamente (pela configuração).

## Troubleshooting (erros comuns e correções)

- Erro: "ModuleNotFoundError: No module named 'mmcv._ext'"
  - Causa: MMCV precisa de extensões compiladas (ops CUDA) que nem sempre estão presentes na instalação pip genérica.
  - Soluções:
    1. Instale uma build do MMCV compatível com sua versão do PyTorch/CUDA. Use `mim install "mmcv>=2.0.0,<2.2.0"` para cu126/torch2.8 conforme mostrado acima.
    2. Para produção, instale `mmcv-full` pré-compilado que corresponde exatamente ao seu CUDA/PyTorch (procure a wheel no site do OpenMMLab) — isso habilita `mmcv._ext`.
    3. Como último recurso temporário (somente para desenvolvimento), é possível usar um stub `mmcv._ext` para evitar crashes de import; porém esse stub não fornece acelerações nem as operações CUDA e deve ser removido em produção.

- Erro: incompatibilidade entre `mmcv`, `mmdet` e `mmpose` (assertions sobre versões)
  - Mantenha os pacotes nas faixas compatíveis: `mmcv>=2.0.0,<2.2.0`; `mmdet>=3.0.0,<3.3.0`; `mmpose` compatível com `mmengine` (o `mim` costuma resolver isso automaticamente).

- Aviso GPU (exemplo RTX 5060 / Ada Lovelace)
  - GPUs muito novas (compute capability >= 12.0) eventualmente não são suportadas por algumas builds oficiais do PyTorch/MMCV e podem emitir avisos do tipo "not compatible with the current PyTorch installation".
  - Se ocorrerem avisos de compatibilidade, duas opções:
    1. Instalar uma build do PyTorch que suporte sua GPU (ex.: PyTorch com suporte a cu128) seguindo https://pytorch.org/get-started/locally/;
    2. Executar no CPU ou em uma GPU compatível até obter uma build apropriada.

## Dicas úteis
- Sempre ative o `venv` antes de rodar os comandos: `source venv/bin/activate`.
- Use `mim install` para gerenciar pacotes do ecossistema OpenMMLab — ele cuida de dependências binárias quando possível.
- Se precisar de uma instalação imutável e reproduzível, considere usar Docker (imagem OpenMMLab) ou um ambiente Conda com canais binários apropriados.

## Contato / Ajuda
Se encontrar um erro que você não consegue resolver, cole o traceback e eu (ou o seu orientador) posso ajudar a diagnosticar a dependência específica.

---

Pequena nota: este README foi atualizado para refletir os passos testados neste repositório (Python 3.12 + PyTorch 2.8.0+cu126) e incluir instruções práticas de resolução de problemas com `mmcv`/`mmcv._ext`.

## Roadmap

### Fase 1: Setup e Preparação de Dados (Semanas 1-2)
- [ ] Download do COCO-WholeBody
- [ ] Conversão para grayscale
- [ ] Implementação de data augmentation
- [ ] Análise exploratória dos dados

### Fase 2: Baseline Model (Semanas 3-4)
- [ ] Setup RTMDet para detecção de pessoas
- [ ] Setup RTMPose para estimação de pose
- [ ] Treinamento baseline com imagens RGB
- [ ] Avaliação baseline

### Fase 3: Adaptação para Grayscale (Semanas 5-7)
- [ ] Fine-tuning RTMDet para grayscale
- [ ] Fine-tuning RTMPose para grayscale
- [ ] Data augmentation específica para IR
- [ ] Avaliação comparativa RGB vs Grayscale

### Fase 4: Otimização e Tempo Real (Semanas 8-10)
- [ ] Otimização de inferência
- [ ] Quantização do modelo
- [ ] Testes de performance
- [ ] Deployment para tempo real

### Fase 5: Validação e Documentação (Semanas 11-12)
- [ ] Testes em ambiente veicular
- [ ] Documentação completa
- [ ] Artigo científico
- [ ] Apresentação TCC

## Métricas de Avaliação

### Object Keypoint Similarity (OKS)
- Métrica padrão do COCO
- Similar ao IoU, mas para keypoints

### PCK (Percentage of Correct Keypoints)
- Porcentagem de keypoints corretamente detectados
- Threshold geralmente em 0.2 (20% da distância torso)

### AP (Average Precision)
- AP@0.5, AP@0.75, AP@0.5:0.95
- Para diferentes thresholds de OKS

### AR (Average Recall)
- AR@0.5, AR@0.75, AR@0.5:0.95

### Tempo de Inferência
- FPS (Frames Per Second)
- Latência média (ms)
- Throughput

## Referências

### Papers
- RTMPose: Real-Time Multi-Person Pose Estimation (2023)
- COCO-WholeBody: COCO with Whole-Body Keypoint Annotations (2020)
- OpenPose: Realtime Multi-Person 2D Pose Estimation (2018)

### Repositories
- MMPose: https://github.com/open-mmlab/mmpose
- RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody

## Contato
- **Autor**: Davi Baechtold Campos
- **Orientador**: Prof. Dr. Alceu de Souza Brito Junior
- **Instituição**: PUCPR
