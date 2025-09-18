# Proposta de Projeto de TCC

Título (provisório): Análise de Lifting de Dados 2D para 3D para Reconhecimento de Gestos

Autor: Davi Baechtold
Data: 2025-09-18

## 1. Motivação e Objetivo

Reconhecer gestos de forma robusta em cenários do mundo real (indústria, veículos, robótica, XR) requer representações espaciais e temporais consistentes. Abordagens 2D (keypoints) são leves e generalistas, mas perdem profundidade e sofrem com oclusões. Este trabalho investiga o lifting 2D→3D (a partir de keypoints 2D) para melhorar a discriminação e a estabilidade temporal no reconhecimento de gestos, explorando também sinais auxiliares (depth monocular e segmentação humana) e cenários multi-view quando possível.

Objetivos:
- O1: Comparar reconhecimento de gestos usando sequências 2D vs. 3D (pós-lifting), mantendo o mesmo classificador temporal.
- O2: Avaliar métricas de estimação 3D (MPJPE, PA-MPJPE, MPVE quando aplicável) e o impacto na acurácia/F1 da classificação de gestos.
- O3: Estudar integração de sinais auxiliares (depth monocular, segmentação humana) e de múltiplas visões (quando disponível) no processo de lifting.
- O4: Demonstrar um protótipo em tempo real com webcam Logitech C922 Pro Stream.

## 2. Escopo e Questões de Pesquisa

- Q1: Em que condições o lifting 2D→3D reduz ambiguidades e melhora a classificação de gestos em relação ao uso apenas de 2D?
- Q2: Qual o ganho de qualidade do lifting com uma única câmera (LiftPose3D-style) vs. múltiplas câmeras (MPL/Transformer multi-view)?
- Q3: Sinais auxiliares (depth monocular, segmentação) ajudam o lifting em cenários desafiadores (oclusões, iluminação, fundo complexo)?
- Q4: Como técnicas de domain adaptation linear (LiftPose3D) facilitam transferir modelos para novos ambientes com poucos dados?

## 3. Metodologia

Pipeline geral:
1) Extração de keypoints 2D (MediaPipe: pose/mãos) de vídeos ou webcam.
2) Lifting 2D→3D:
   - Monocular (single-view): baseline LiftPose3D-like (MLP/TCN) sobre joints 2D normalizados.
   - Multi-view (opcional): fusão via Transformer (MPL-like) de esqueleto 2D por câmera para um único esqueleto 3D.
3) Enriquecimento opcional do espaço latente com:
   - Depth monocular por frame (ex.: MiDaS/Torch Hub) resumido por estatísticas regionais alinhadas ao esqueleto.
   - Segmentação humana (máscara binária/contornos) para robustez a fundos complexos.
4) Classificação temporal de gestos (Transformer encoder; baselines LSTM/MLP) sobre sequências 2D vs. 3D vs. 3D+auxiliares.
5) Avaliação offline (manifests) e em tempo real (webcam C922).

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

## 6. Protocolo Experimental

- P1: Treinar classificador com 2D (baseline atual do repositório); avaliar em `manifest_val.csv` e `manifest_test.csv`.
- P2: Treinar lifter 2D→3D com MPJPE em dados sintéticos ou pares 2D↔3D de dataset público; converter sequências 2D para 3D; re-treinar classificador no 3D.
- P3: Ablations: 2D vs. 3D vs. 3D+depth/seg; LSTM vs. Transformer; normalização por-junta vs. global.
- P4: Domain adaptation linear (LiftPose3D) para pequena coleta própria (C922) e comparação.
- P5: Demo tempo real com pipeline on-device; medir latência e robustez.

Critérios de sucesso:
- +ΔF1 macro de X% ao migrar 2D→3D em pelo menos N classes.
- MPJPE ≤ baseline simples do lifter monocular; redução adicional sob PA-MPJPE.
- Latência do demo ≤ 60 ms/frame em CPU (objetivo; ajustar conforme hardware).

## 7. Implementação no Repositório (Roadmap)

- `scripts/`:
  - `train_lifter.py`: treinar MLP/TCN 2D→3D com MPJPE.
  - `lift_sequences.py`: converter `.npz` 2D em `.npz` 3D.
  - `evaluate_3d_metrics.py`: calcular MPJPE/PA-MPJPE (e MPVE se aplicável).
  - `depth_segment_features.py` (opcional): extrair features de depth/segmentação alinhadas aos joints.
- `src/models/lifter.py`, `src/models/temporal_tcn.py`.
- `configs/lifter.yaml`.

Integração mínima ao README: link para esta proposta e checklist de experimentos.

## 8. Riscos, Limitações e Ética

- Ambiguidade de profundidade em monocular; dependência de normalização e qualidade 2D.
- Datasets com bias de domínio; validar com domain adaptation.
- Privacidade: evitar armazenar vídeos crus; preferir sequências de keypoints/mascaras.
- Uso de CPU: priorizar modelos leves para demo.

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
