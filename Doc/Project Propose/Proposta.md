# Proposta de Projeto de TCC

**Título**: 2D-to-3D Lifting for In-Cabin Occupant Posture Monitoring: A Comparative Analysis of Multi-View and Monocular Depth Fusion Approaches

**Autor**: Davi Baechtold Campos  
**Orientador**: Prof. Dr. Alceu de Souza Brito Junior  
**Banca**: Prof. Joed Zimmer, Prof. Alessandro Zimmer  
**Data**: Setembro 2025

---

> **📝 Contexto**: Esta proposta reflete o direcionamento estratégico definido na reunião de alinhamento, onde o foco evoluiu do reconhecimento de gestos genérico para o **problema fundamental do 2D-to-3D lifting** em ambientes com oclusões severas. O objetivo é produzir uma **contribuição científica mensurável** com potencial para publicação.

## 1. Motivação e Objetivo

### Contexto: Monitoramento de Ocupantes em Cabine Veicular

O monitoramento preciso da postura de ocupantes em cabines veiculares é fundamental para sistemas de segurança, detecção de sonolência e interfaces gestuais avançadas. **O principal desafio técnico** reside na reconstrução 3D robusta em ambientes com severas oclusões (volante, painel, bancos) e condições de iluminação variáveis.

Abordagens tradicionais baseadas apenas em keypoints 2D são insuficientes para este cenário, pois perdem informação crítica de profundidade e sofrem com ambiguidades espaciais. Este trabalho foca no problema fundamental: **como obter estimação de pose 3D precisa e robusta** que serve como base para qualquer aplicação subsequente de reconhecimento de gestos.

### Objetivos Científicos

**Principal**: Realizar análise comparativa quantitativa de técnicas de 2D-to-3D lifting para determinação da abordagem mais eficaz em ambiente veicular.

**Específicos**:
- **O1**: Comparar precisão de lifting monocular vs. multi-view em cenários com oclusão
- **O2**: Quantificar ganho de performance com fusão de profundidade monocular 
- **O3**: Avaliar métricas de estimação 3D (MPJPE, PA-MPJPE) e impacto na classificação de gestos
- **O4**: Desenvolver pipeline em tempo real para validação prática (webcam C922)

### Potencial de Publicação

Este trabalho visa contribuição científica original através da **análise metódica de diferentes soluções técnicas** para um problema específico e relevante. Os resultados quantitativos (ex.: "fusão com profundidade monocular reduz erro de posicionamento em X% em ambiente veicular") constituem contribuição publicável para a área.

## 2. Questões de Pesquisa e Hipóteses

### Questões Centrais

**Q1**: **Oclusões Severas** - Qual a degradação de precisão do lifting 2D→3D sob oclusões típicas de cabine veicular (volante, painel, bancos)?

**Q2**: **Multi-view vs. Monocular** - Quantitativamente, qual o ganho de precisão ao usar múltiplas câmeras sincronizadas vs. uma única câmera com fusão de profundidade?

**Q3**: **Profundidade como Proxy** - A informação de profundidade monocular consegue compensar parcialmente a ausência de múltiplas vistas?

**Q4**: **Generalização** - Como técnicas de domain adaptation (LiftPose3D) facilitam transferência para o ambiente veicular com dados limitados?

### Hipóteses de Trabalho

- **H1**: Multi-view reduz significativamente ambiguidades espaciais (>20% melhoria em MPJPE)
- **H2**: Fusão com profundidade monocular oferece ganho intermediário (10-15% melhoria)  
- **H3**: Qualidade do lifting 3D correlaciona diretamente com precisão de classificação gestual
- **H4**: Pipeline em tempo real é viável com latência <60ms em hardware consumer

## 3. Metodologia

### Pipeline Comparativo

**Foco**: Análise quantitativa de precisão de lifting (não aplicação final de reconhecimento)

**Etapas**:
1. **Extração 2D**: MediaPipe Holistic (pose + mãos) com tratamento de oclusões
2. **Lifting Comparativo**:
   - **Baseline**: Single-view LiftPose3D-style (MLP/TCN)  
   - **Multi-view**: Fusão Transformer (MPL-inspired) para múltiplas câmeras
   - **Depth-Enhanced**: Single-view + fusão profundidade monocular
3. **Validação 3D**: Métricas MPJPE/PA-MPJPE para análise quantitativa
4. **Prova de Conceito**: Classificação gestual para validar utilidade prática
5. **Demo Real-time**: Pipeline otimizado com webcam C922

### Dados Sintéticos e Reais

**Sintéticos**: Pipeline AMASS + renderização multi-view para ground-truth 3D controlado  
**Reais**: Datasets públicos + coleta própria para domain adaptation  
**Foco Veicular**: Simulação de oclusões e constraints típicos de cabine

### 3.1. Lifting Single-View (LiftPose3D)
- Entrada: joints 2D normalizados por root e escala (e.g., centro do quadril + normalização por distância ombro).
- Modelo: MLP/TCN com camadas residuais, perda MPJPE; variante PA-MPJPE para análise pós-alinhamento.
- Domain Adaptation: mapeamento linear de poses 2D para reduzir shift entre dataset fonte (pré-treino) e domínio alvo (webcam).

### 3.2. Lifting Multi-View (MPL-like)
- Pré-requisito: múltiplas câmeras sincronizadas (quando disponível em dataset público). Não é exigência do demo com webcam.
- Estratégia: estimar pose 2D por vista; empilhar e fundir via Transformer; supervisionar com 3D GT (real ou sintético).
- Dados Sintéticos: renderização de malhas (AMASS) para gerar pares 2D ruidosos ↔ 3D, como em MPL.

### 3.3. Depth Monocular e Segmentação Humana
- Depth: usar um estimador leve (MiDaS pequeno) em CPU, extrair features por regiões (por junta, média/local patch) para compor vetores auxiliares por frame.
- Segmentação: máscara humana para filtrar ruídos do fundo e estabilizar detecção 2D.

## 4. Métricas

Estimação 3D:
- MPJPE: média da distância euclidiana por articulação.
- PA-MPJPE: MPJPE após alinhamento de Procrustes (translação/rotação/escala).
- MPVE: média por vértice em malha (se usarmos malhas; caso contrário, opcional).

Classificação de Gestos:
- Acurácia, F1 macro, matriz de confusão, curva PR por classe.
- Latência média (ms/frame) no demo em tempo real (C922, CPU).

## 5. Datasets e Coleta

- Públicos: 20BN-Jester (gestos manuais), outros se necessário (e.g., SHREC, NTU RGB+D apenas como referência de protocolo).
- Coleta própria (opcional): conjunto pequeno de gestos alvo gravados com a C922 para avaliar domain adaptation.
- Manifestos `.csv` no formato `path,label` apontando para `.npz` com sequências de keypoints (2D) e, quando disponível, variantes com 3D.

## 6. Protocolo Experimental e Cronograma

### Fases de Desenvolvimento

| Fase | Entrega | Duração | Modalidades |
|------|---------|---------|-------------|
| **A** | Baseline 3D lifter | 3 semanas | keypoints 2D→3D |
| **B** | Multi-view pipeline | 4 semanas | múltiplas câmeras |  
| **C** | Depth fusion | 3 semanas | + profundidade monocular |
| **D** | Análise comparativa | 2 semanas | métricas quantitativas |
| **E** | Demo real-time | 2 semanas | validação prática |
| **F** | Documentação final | 2 semanas | artigo + relatório |

### Critérios de Sucesso Quantitativos

**Técnicos**:
- Multi-view: **MPJPE < 80mm** vs. baseline monocular >100mm  
- Depth fusion: **MPJPE < 90mm** (ganho intermediário)
- Real-time: **Latência ≤ 60ms/frame** em hardware consumer

**Científicos**:  
- **Contribuição mensurável**: quantificação precisa de trade-offs entre abordagens
- **Reprodutibilidade**: código e dados sintéticos disponibilizados
- **Publicabilidade**: resultados com significância estatística e relevância prática


## 7. Impacto e Aplicações

### Contribuição Científica
- **Análise comparativa rigorosa** de técnicas state-of-the-art em contexto específico
- **Quantificação de trade-offs** entre precisão, complexidade e custo computacional  
- **Metodologia replicável** para avaliação de lifting em ambientes com restrições

### Aplicações Práticas
- **Automotiva**: sistemas avançados de assistência ao motorista (ADAS)
- **Segurança**: detecção de sonolência e postura inadequada
- **HMI**: interfaces gestuais naturais para controle veicular
- **Robótica**: monitoramento de ocupantes em veículos autônomos

## 8. Riscos e Limitações

### Técnicas
- **Ambiguidade monocular**: inerente à reconstrução 3D de vista única
- **Qualidade 2D**: dependência crítica da detecção de keypoints robusta  
- **Domain gap**: diferenças entre dados sintéticos e cenários reais

### Mitigação
- **Dados sintéticos controlados**: AMASS para ground-truth preciso
- **Domain adaptation**: técnicas lineares para transferência
- **Validação múltipla**: datasets públicos + coleta própria

## 9. Referências (anexas na pasta Doc/References/2D to 3D)
- LiftPose3D: mapeamento 2D→3D com domain adaptation linear.
- MPL (Multi-view Pose Lifter): fusão Transformer com dados sintéticos (AMASS).
- Survey “An In-Depth Analysis of 2D and 3D Pose Estimation Techniques”.

## 10. Estado da Arte (Resumo)

- Aplicações: HCI/XR, robótica colaborativa, direção assistida por gestos, segurança ocupacional e análise de movimento. O reconhecimento baseado em pose é leve, interpretável e menos sensível a variações de aparência.
- Vídeo/Temporal: além de MLP, modelos como RNN/LSTM, TCN e Transformers temporais capturam dependências de longo alcance. TCNs e Transformers têm mostrado excelente desempenho para sequências de keypoints.
- Cinético (MoCap) vs. Infravermelho/RGB: MoCap é padrão-ouro para GT 3D, porém caro e restrito; RGB/IR são acessíveis, e o lifting permite recuperar 3D apenas com câmeras comuns.
- Modelos pré-treinados (Língua de Sinais/Gestos): uso de keypoints 2D/3D como entrada facilita transferência entre domínios; domain adaptation linear (LiftPose3D) reduz necessidade de grande re-treinamento.
- Pose 2D: pipelines top-down (detecção→pose) vs. bottom-up (chaves→agrupamento), com trade-offs de precisão e velocidade; MediaPipe oferece solução prática para tempo real.
- Pose 3D (Lifting): single-view é ambíguo mas efetivo com normalização e perdas adequadas; multi-view baseado em Transformer (MPL) resolve oclusões e melhora a precisão, inclusive com dados sintéticos (AMASS).
