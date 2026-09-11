# Estrutura do Projeto - TCC

**Última atualização**: 19 de Outubro de 2025

## 📁 Estrutura de Diretórios

```
Project/
├── checkpoints/                      # Checkpoints pré-treinados
│   ├── rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
│   └── rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
│
├── configs/                          # Arquivos de configuração
│   ├── rtmpose_m_wholebody.py        # Config principal (completa)
│   ├── rtmpose_m_wholebody_minimal.py # Config rápida para testes
│   └── detectors/
│       └── rtmdet_nano_person_infer.py
│
├── data/                             # Datasets (excluído do git)
│   ├── raw/                          # COCO-WholeBody RGB original
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── annotations/
│   ├── processed/grayscale/          # Imagens convertidas + augmentation
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── annotations/
│   └── video/                        # Vídeos de teste
│
├── src/                              # Código fonte
│   ├── data/
│   │   ├── download_coco.py          # Download do COCO-WholeBody
│   │   ├── convert_to_gray.py        # Conversão RGB → Grayscale
│   │   └── augmentation.py           # Data augmentation
│   ├── training/
│   │   └── train_pose.py             # Script de treinamento
│   └── evaluation/
│       ├── evaluate_pose.py          # Avaliação RGB vs Grayscale
│       └── run_realtime.py           # Inferência em tempo real
│
├── scripts/                          # Scripts auxiliares
│   ├── prepare_dataset.sh            # Preparação automática do dataset
│   ├── plot_training.py              # Visualização de curvas de treino
│   └── train_full_pipeline.sh        # Pipeline completo de treinamento
│
├── work_dirs/                        # Outputs de treinamento
│   ├── test_minimal5/                # ⭐ Melhor modelo atual
│   │   ├── best_coco-wholebody_AP_epoch_50.pth
│   │   ├── rtmpose_m_wholebody_minimal.py
│   │   └── 20251007_201644/          # Logs de treinamento
│   ├── eval_results/                 # Resultados de avaliação
│   │   ├── rgb/
│   │   └── ir/
│   └── video_eval/                   # Avaliações em vídeo
│
├── plots/                            # Gráficos e visualizações
├── docs/                             # Documentação adicional
├── venv/                             # Ambiente virtual Python
│
├── README.md                         # Documentação principal
├── requirements.txt                  # Dependências Python
├── cleanup_project.sh                # Script de limpeza
└── .gitignore                        # Arquivos ignorados pelo Git
```

## 📊 Tamanhos Aproximados

| Diretório | Tamanho Aproximado | Observação |
|-----------|-------------------|------------|
| `data/` | ~40-50 GB | Excluído do Git |
| `checkpoints/` | ~200 MB | Checkpoints pré-treinados |
| `work_dirs/test_minimal5/` | ~500 MB | Melhor modelo + logs |
| `venv/` | ~2-3 GB | Ambiente Python |
| `src/` | ~100 KB | Código fonte |

## 🗑️ Arquivos Removidos na Limpeza

### Removidos permanentemente:
- `GRAYSCALE_RT_PLAN.md` - Documento duplicado
- `configs/rtmpose_m_wholebody_ultra_minimal.py` - Config não usado
- `configs/rtmpose_s_grayscale_rt.py` - Config não usado
- `work_dirs/baseline_grayscale/` - Experimentos antigos
- `work_dirs/finetune_grayscale/` - Experimentos antigos
- `rtmpose-l_simcc-*.pth` - Checkpoint não organizado
- `rtmw-x_simcc-*.pth` - Checkpoint não organizado
- `test_train_minimal.py` - Script de teste antigo
- `run_evaluation.sh` - Script não usado
- `Project/` - Pasta duplicada

### Mantidos como backup:
- `README.md.backup` - Versão anterior do README

## 🎯 Arquivos Principais

### Para Usuários:
1. **README.md** - Guia completo de instalação e uso
2. **requirements.txt** - Lista de dependências
3. **scripts/prepare_dataset.sh** - Preparação automática

### Para Desenvolvimento:
1. **src/training/train_pose.py** - Script de treinamento
2. **src/evaluation/run_realtime.py** - Inferência em tempo real
3. **configs/rtmpose_m_wholebody*.py** - Configurações

### Outputs:
1. **work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth** - Melhor modelo
2. **work_dirs/eval_results/** - Resultados de avaliação

## 🔧 Manutenção

Para manter o projeto limpo:

```bash
# Executar script de limpeza
./cleanup_project.sh

# Limpar apenas caches Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Verificar tamanho dos diretórios
du -h --max-depth=1 | sort -h
```

## 📝 Notas

- O diretório `data/` deve ser baixado separadamente (veja README.md)
- O diretório `venv/` deve ser criado localmente (não está no Git)
- Checkpoints pré-treinados devem ser baixados manualmente
- Logs de treinamento estão em `work_dirs/test_minimal5/20251007_201644/`

---

**Status**: Projeto limpo e organizado (19/10/2025)
