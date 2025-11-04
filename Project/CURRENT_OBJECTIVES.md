# 🎯 Objetivos Atualizados do Projeto - Novembro 2025

**Data**: Novembro 3, 2025  
**Atualização baseada em**: Feedback do usuário

---

## 📋 Objetivos Definidos para Implementação

### 1. ✅ **Multi-Pessoa com Top-Down (RTMDet + RTMPose)**
**Prioridade**: ✅ **JÁ IMPLEMENTADO - TESTAR**

**Status**: O código já está pronto! Precisa apenas validar que funciona.

**Comando para testar**:
```bash
cd /home/davs/Documents/TCC/Project
source venv/bin/activate

# Multi-pessoa com detector RTMDet
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

**O que faz**:
1. RTMDet detecta todas as pessoas na imagem → bounding boxes
2. Para cada bounding box, RTMPose extrai 133 keypoints
3. Visualiza skeleton colorido para cada pessoa
4. Funciona em webcam ou vídeo

**Performance esperada**: ~25-30 FPS para 2-3 pessoas (RTX 5060)

---

### 2. 🔴 **Extração 2D → 3D (Plano Cartesiano XYZ)**
**Prioridade**: 🔴 **ALTA - A IMPLEMENTAR**

**Objetivo**: Converter keypoints 2D (x, y) para coordenadas 3D (x, y, z).

**Solução Proposta**: MLP simples para lifting (LiftPose3D-style)

#### Implementação

**Arquivo novo**: `src/models/lifting/simple_lifter.py`

Veja código completo em `IMPLEMENTATION_PLAN.md` (já existe um plano detalhado).

**Dataset necessário**: Human3.6M
- Registre em: http://vision.imar.ro/human3.6m/
- Download: ~100GB (subsets S1, S5, S6, S7, S8 para treino)
- Ground truth 3D disponível

**Tempo estimado**: 2-3 semanas
- 1 semana: setup + implementação do lifter
- 1-2 semanas: treinamento + validação

---

### 3. 🟡 **Dataset Drive&Act para Aplicação Veicular**
**Prioridade**: 🟡 **MÉDIA - A IMPLEMENTAR**

**Objetivo**: Validar modelo em cenário veicular real (motoristas, passageiros).

#### Dataset Drive&Act
- **Descrição**: 15h de vídeo, 6 câmeras, ambiente veicular
- **Anotações**: Atividades, bounding boxes, contexto veicular
- **Tamanho**: ~200GB
- **Licença**: Acadêmica (requer registro)
- **Download**: https://driveandact.com/

#### Pipeline de Integração

**Passo 1**: Download e conversão para COCO format
**Passo 2**: Fine-tuning do modelo RTMPose em Drive&Act
**Passo 3**: Avaliação de performance em oclusões veiculares (volante, painel)

**Script proposto**: `src/data/convert_driveact_to_coco.py` (ver `IMPLEMENTATION_PLAN.md`)

**Tempo estimado**: 2-3 semanas
- 1 semana: download + conversão
- 1-2 semanas: fine-tuning + avaliação

---

### 4. 🟡 **Melhorar Treinamento para Maior Precisão**
**Prioridade**: 🟡 **MÉDIA - MELHORAR GRADUALMENTE**

**Status Atual**: AP = 0.4373 (bom, mas pode melhorar)
**Meta**: AP > 0.55 (+25% de ganho)

#### Estratégias de Melhoria

**A. Treinar Mais Epochs** ⏱️ 2 semanas
- Atual: 50 epochs
- Target: 270 epochs (como RTMPose oficial)
- Ganho esperado: +5-10% AP

**B. Usar Modelo Maior** 💾 2-3 semanas
- Atual: RTMPose-M (768 channels)
- Target: RTMPose-L (1024 channels)
- Ganho esperado: +8-12% AP

**C. Aumentar Resolução** 🖼️ 1-2 semanas
- Atual: 256×192
- Target: 384×288
- Ganho esperado: +5-8% AP

**D. Ajustar Learning Rate** 🎓 1 semana
- Warmup mais longo
- Learning rate decay otimizado
- Ganho esperado: +2-4% AP

**E. Data Augmentation Otimizada** 🔧 1-2 semanas
- Adicionar: CutOut, MixUp, Oclusão sintética
- Simular: volante, painel (para Drive&Act)
- Ganho esperado: +3-5% AP

---

### 5. 📝 **Documentação Completa**
**Prioridade**: 🟢 **BAIXA - FAZER GRADUALMENTE**

#### A. Review of 2D Keypoint Metrics ✅ PARCIALMENTE FEITO

**Onde está**: `EVALUATION_GUIDE.md`

**O que tem**:
- ✅ Explicação de AP, AR, AP.5, AP.75
- ✅ Tabela de métricas do RTMPose oficial
- ✅ Comparação com seu modelo

**O que falta**:
- ❌ Métricas por região anatômica (body, face, hands, feet)
- ❌ Análise de erro por tipo de oclusão
- ❌ Gráficos de curvas PR

**Arquivo a criar**: `docs/METRICS_ANALYSIS.md`

#### B. Dataset Documentation ✅ PARCIALMENTE FEITO

**Onde está**: `README.md` (seção Dataset)

**O que tem**:
- ✅ COCO-WholeBody: 133 keypoints breakdown
- ✅ Conversão RGB → Grayscale explicada
- ✅ Estrutura de diretórios

**O que falta**:
- ❌ Human3.6M (para 3D lifting)
- ❌ Drive&Act (para aplicação veicular)
- ❌ Estatísticas detalhadas (distribuição de poses, etc.)

**Arquivo a criar**: `docs/DATASETS.md`

#### C. Data Augmentation Documentation ✅ FEITO

**Onde está**: `README.md` (seção Data Augmentation)

**O que tem**:
- ✅ 6 técnicas explicadas (vignetting, noise, blur, contrast, brightness, rotation/flip)
- ✅ Motivação científica para cada técnica
- ✅ Relação com câmeras IR

**Status**: ✅ **COMPLETO**

#### D. Architecture Documentation ❌ NÃO FEITO

**O que falta**:
- ❌ Descrição detalhada do RTMPose-M
- ❌ Backbone: CSPNeXt architecture
- ❌ Head: RTMCCHead + SimCC decoder
- ❌ Training configuration completa
- ❌ Diagrama de arquitetura

**Arquivo a criar**: `docs/ARCHITECTURE.md`

---

## 📊 Priorização e Timeline

### Cronograma Proposto (12 semanas)

#### Fase 1: Validação e Setup (Semanas 1-2)
| Tarefa | Tempo | Prioridade |
|--------|-------|------------|
| ✅ Testar multi-pessoa com RTMDet | 1 dia | 🔴 Alta |
| Download Human3.6M dataset | 2 dias | 🔴 Alta |
| Setup ambiente para 3D lifting | 2 dias | 🔴 Alta |
| Documentar arquitetura atual | 3 dias | 🟢 Baixa |

#### Fase 2: 3D Lifting Implementation (Semanas 3-5)
| Tarefa | Tempo | Prioridade |
|--------|-------|------------|
| Implementar SimplePoseLifter (MLP) | 1 semana | 🔴 Alta |
| Treinar lifter em Human3.6M | 1 semana | 🔴 Alta |
| Validar MPJPE < 100mm | 3 dias | 🔴 Alta |
| Integrar no pipeline de inferência | 2 dias | 🔴 Alta |

#### Fase 3: Dataset Drive&Act (Semanas 6-8)
| Tarefa | Tempo | Prioridade |
|--------|-------|------------|
| Download Drive&Act | 2 dias | 🟡 Média |
| Converter para COCO format | 3 dias | 🟡 Média |
| Fine-tune modelo RTMPose | 1 semana | 🟡 Média |
| Avaliar performance veicular | 3 dias | 🟡 Média |

#### Fase 4: Otimização e Documentação (Semanas 9-12)
| Tarefa | Tempo | Prioridade |
|--------|-------|------------|
| Treinar 270 epochs (AP > 0.55) | 2 semanas | 🟡 Média |
| Documentação completa (métricas, datasets) | 1 semana | 🟢 Baixa |
| Criar demos e visualizações | 3 dias | 🟢 Baixa |
| Preparar apresentação final | 2 dias | 🟢 Baixa |

---

## ✅ Checklist de Ações Imediatas

### Esta Semana (Novembro 4-8, 2025)

- [ ] **Dia 1**: Testar multi-pessoa com webcam (comando acima)
  - Confirmar que funciona
  - Medir FPS real
  - Testar com 2-3 pessoas

- [ ] **Dia 2**: Registrar em Human3.6M
  - Acessar http://vision.imar.ro/human3.6m/
  - Preencher formulário acadêmico
  - Aguardar aprovação (1-2 dias)

- [ ] **Dia 3**: Implementar SimplePoseLifter
  - Criar arquivo `src/models/lifting/simple_lifter.py`
  - Copiar código do plano
  - Testar forward pass

- [ ] **Dia 4-5**: Preparar pipeline de treinamento 3D
  - Script de carregamento Human3.6M
  - Loss function (MPJPE)
  - Training loop básico

---

## 📞 Dúvidas Frequentes

### Q: Multi-pessoa já funciona?
**A**: Sim! O código está implementado. Rode o comando da seção 1 para testar.

### Q: Preciso implementar bottom-up também?
**A**: Não necessário. Top-down com RTMDet é suficiente e mais preciso.

### Q: Quanto tempo leva para treinar 3D lifting?
**A**: ~3-5 dias em RTX 5060 (50 epochs, ~6h por epoch).

### Q: Drive&Act é obrigatório?
**A**: Não, mas é importante para validar aplicação veicular. Pode ser opcional se o tempo for curto.

### Q: Posso usar outro dataset 3D além de Human3.6M?
**A**: Sim! Alternativas:
- MPI-INF-3DHP (indoor/outdoor, mais diverso)
- CMU Panoptic (multi-view, maior)
- AMASS (sintético, mais fácil acesso)

---

## 📚 Arquivos de Referência

| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `README.md` | Documentação principal | ✅ Completo |
| `EVALUATION_GUIDE.md` | Guia de avaliação com métricas | ✅ Completo |
| `IMPLEMENTATION_PLAN.md` | Plano detalhado (antigo) | ✅ Existe |
| `ROADMAP.md` | Roadmap geral | ✅ Existe |
| `script.txt` | Comandos práticos | ✅ Atualizado |
| `docs/DATASETS.md` | Documentação de datasets | ❌ A criar |
| `docs/ARCHITECTURE.md` | Documentação de arquitetura | ❌ A criar |
| `docs/METRICS_ANALYSIS.md` | Análise de métricas | ❌ A criar |

---

## 🚀 Próximo Passo AGORA

**Execute este comando para testar multi-pessoa**:

```bash
cd /home/davs/Documents/TCC/Project
source venv/bin/activate

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

**Observações ao testar**:
- Fique na frente da webcam
- Peça alguém para entrar no frame (testar multi-pessoa)
- Anote o FPS mostrado no canto da tela
- Veja se os skeletons são desenhados corretamente
- Pressione 'q' para sair

**Depois de testar**, reporte:
- ✅ Funcionou?
- Quantas pessoas conseguiu detectar simultaneamente?
- Qual foi o FPS?

---

**Última atualização**: Novembro 3, 2025  
**Status**: Aguardando teste de multi-pessoa + início de 3D lifting
