# 📊 Guia de Avaliação do Modelo - Métricas Simplificadas

Este documento explica como avaliar o modelo treinado usando as métricas oficiais do COCO-WholeBody (AP, AR), no mesmo formato da documentação do RTMPose.

---

## 🎯 Tipos de Avaliação

### 1. **Avaliação Completa com Métricas COCO** (Recomendado)

Usa o evaluator oficial do MMPose para calcular todas as métricas COCO-WholeBody:
- Whole AP, Whole AR
- AP/AR .5, .75
- AP/AR (M), (L)  
- Métricas por região (body, face, hands, feet)

**Comando**:
```bash
cd /home/davs/Documents/TCC/Project

# Ative o ambiente virtual primeiro
source venv/bin/activate

# Rode a avaliação
python src/evaluation/evaluate_simple.py \
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
  --device cuda:0
```

**Output Esperado**:
```
==================================================
📊 EVALUATION RESULTS
==================================================

Metric                         Value          
--------------------------------------------------
Whole AP                       0.4373
Whole AP .5                    0.7683
Whole AP .75                   0.4442
Whole AP (M)                   0.4653
Whole AP (L)                   0.4379
Whole AR                       0.5287
Whole AR .5                    0.8054
Whole AR .75                   0.5658
Whole AR (M)                   0.5276
Whole AR (L)                   0.5329
==================================================
```

---

### 2. **Comparação Visual RGB vs. Grayscale**

Compara visualmente as predições em imagens RGB vs. Grayscale (útil para debug):

**Comando**:
```bash
python src/evaluation/evaluate_pose.py \
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
  --rgb-dir data/raw/val2017 \
  --ir-dir data/processed/grayscale/val2017 \
  --out-dir work_dirs/eval_results \
  --n 20
```

**Output**: Gera visualizações em `work_dirs/eval_results/{rgb,ir}/`

---

## 📋 Métricas COCO-WholeBody Explicadas

### Principais Métricas

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **Whole AP** | Average Precision (geral) | Média da precisão em diferentes thresholds IoU (0.5:0.95) |
| **Whole AP .5** | AP em IoU=0.5 | Precisão com threshold relaxado (mais fácil) |
| **Whole AP .75** | AP em IoU=0.75 | Precisão com threshold rigoroso (mais difícil) |
| **Whole AP (M)** | AP para pessoas médias | Área 32²-96² pixels |
| **Whole AP (L)** | AP para pessoas grandes | Área >96² pixels |
| **Whole AR** | Average Recall (geral) | Proporção de keypoints detectados corretamente |
| **Whole AR .5** | AR em IoU=0.5 | Recall com threshold relaxado |
| **Whole AR .75** | AR em IoU=0.75 | Recall com threshold rigoroso |

### Como Interpretar

- **AP > 0.40**: Bom desempenho
- **AP > 0.50**: Muito bom desempenho
- **AP > 0.60**: Excelente desempenho
- **AP > 0.70**: Estado da arte

**Seu modelo atual**: AP = 0.4373 (Bom desempenho! 🎉)

---

## 🚀 Inferência em Tempo Real

### Webcam (Single Person)
```bash
python src/evaluation/run_realtime.py \
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
  --device cuda:0 \
  --source 0
```

### Webcam (Multi-Person com Detector)
```bash
python src/evaluation/run_realtime.py \
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
  --det-cfg configs/detectors/rtmdet_nano_person_infer.py \
  --det-ckpt checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
  --bbox-thr 0.5 \
  --score-thr 0.4 \
  --device cuda:0 \
  --source 0
```

### Vídeo
```bash
python src/evaluation/run_realtime.py \
  --cfg work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py \
  --ckpt work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth \
  --device cuda:0 \
  --source data/video/seu_video.mp4
```

**Controles**:
- Pressione `q` para sair
- FPS exibido no canto superior esquerdo

---

## 🔍 Comparação com RTMPose Oficial

### Resultados Oficiais do RTMPose (COCO-WholeBody)

| Config | Input Size | Whole AP | Whole AR | FLOPS (G) |
|--------|-----------|----------|----------|-----------|
| RTMW-m | 256x192 | 58.2 | 67.3 | 4.3 |
| RTMW-l | 256x192 | 66.0 | 74.6 | 7.9 |
| RTMW-x | 256x192 | 67.2 | 75.2 | 13.1 |
| RTMW-l | 384x288 | 70.1 | 78.0 | 17.7 |
| RTMW-x | 384x288 | 70.2 | 78.1 | 29.3 |

**Fonte**: [RTMPose GitHub](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#wholebody-2d-133-keypoints)

### Seu Modelo

| Config | Input Size | Whole AP | Whole AR | Dataset |
|--------|-----------|----------|----------|---------|
| RTMPose-M (seu) | 256x192 | **0.4373** | **0.5287** | COCO-WB Grayscale (50 epochs) |

**Análise**:
- ✅ Seu modelo foi treinado com apenas **50 epochs** (vs. 270-420 do oficial)
- ✅ Dataset foi **convertido para grayscale** (maior dificuldade)
- ✅ AP 0.4373 é **razoável** considerando as condições
- 🎯 Com treinamento completo (270+ epochs) e fine-tuning, pode chegar perto de 0.58

---

## 📈 Melhorias Possíveis

### Curto Prazo (2-4 semanas)
1. **Treinar mais epochs**: 50 → 270 epochs
2. **Usar modelo maior**: RTMPose-M → RTMPose-L ou X
3. **Aumentar resolução**: 256x192 → 384x288
4. **Ajustar augmentations**: Testar diferentes combinações

### Médio Prazo (1-2 meses)
1. **Fine-tuning seletivo**: Descongelar apenas cabeça primeiro
2. **Learning rate warmup**: Ajustar scheduler
3. **Mix precision training**: Usar AMP para treinar mais rápido
4. **Ensemble**: Combinar múltiplos modelos

### Longo Prazo (2-3 meses)
1. **Implementar 3D lifting**: 2D → 3D pose estimation
2. **Multi-view fusion**: Usar múltiplas câmeras
3. **Depth integration**: Fusão com profundidade monocular
4. **Dataset veicular**: Treinar em Drive&Act

---

## 🐛 Troubleshooting

### Erro: "No module named 'mmpose'"
```bash
# Ative o ambiente virtual
cd /home/davs/Documents/TCC/Project
source venv/bin/activate

# Instale dependências
pip install mmpose mmengine mmcv
```

### Erro: "CUDA out of memory"
```bash
# Reduza o batch size no config
# Edite: configs/rtmpose_m_wholebody_minimal.py
# Mude: batch_size=32 → batch_size=16 ou batch_size=8
```

### Erro: "venv/bin/activate: No such file or directory"
```bash
# Crie novo ambiente virtual
cd /home/davs/Documents/TCC/Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📚 Referências

- **RTMPose Paper**: [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
- **COCO-WholeBody Dataset**: [GitHub](https://github.com/jin-s13/COCO-WholeBody)
- **MMPose Documentation**: [ReadTheDocs](https://mmpose.readthedocs.io/)
- **COCO Metrics**: [cocodataset.org](https://cocodataset.org/#keypoints-eval)

---

**Última atualização**: Novembro 2025
