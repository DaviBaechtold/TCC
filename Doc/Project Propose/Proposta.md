# Proposta de Projeto de TCC

Título (provisório): Lifting Multimodal 2D→3D com Espaço Latente Integrando Depth Monocular, Segmentação Humana, Multi‑View e Video Embeddings para Reconhecimento de Gestos

Autor: Davi Baechtold  
Data: 2025-09-22

## 1. Motivação e Objetivo

Modelos de reconhecimento de gestos baseados apenas em keypoints 2D sofrem com ambiguidade de profundidade, oclusões e perda de contexto dinâmico. Avanços recentes em: 
1) lifting 2D→3D (reduzindo ambiguidades geométricas), 
2) depth monocular (Depth Anything 2 / Depth Pro), 
3) segmentação humana robusta, e 
4) embeddings de vídeo pré‑treinados (capturam dinâmica e textura global), 
permitem compor um espaço latente multimodal mais informativo. Com múltiplas câmeras (quando disponíveis), a fusão multi-view pode ainda reduzir incerteza estrutural. Este projeto propõe uma arquitetura unificada que gera e explora esse espaço latente para melhorar lifting e, por consequência, a classificação de gestos.

Objetivos Específicos:
- O1: Projetar e implementar um modelo multimodal de lifting 2D→3D que integre keypoints + depth monocular + máscara de segmentação + embeddings de vídeo.
- O2: Investigar a contribuição incremental de cada modalidade para a qualidade 3D (MPJPE / PA-MPJPE) e para métricas de classificação de gestos (F1 macro, acurácia).
- O3: Extender a arquitetura para multi‑view quando um dataset com múltiplas câmeras estiver disponível, analisando ganhos versus single-view enriquecido.
- O4: Demonstrar um pipeline (quase) em tempo real (webcam) usando depth monocular leve como proxy.
- O5: Criar protocolo de ablação e documentação reprodutível no repositório.

## 2. Escopo e Questões de Pesquisa

Perguntas:
- Q1: Em que medida depth monocular e segmentação reduzem erros de lifting em cenários com oclusões parciais?
- Q2: Video embeddings (3D CNN / ViT temporal) acrescentam sinal complementar além de keypoints + depth? Em quais gestos (rápidos vs. estáticos)?
- Q3: Multi‑view oferece ganho substancial adicional sobre single‑view multimodal ou os proxies (depth + video) já capturam a maior parte do benefício?
- Q4: Qual a sensibilidade do espaço latente a ruído em keypoints (ex.: jitter de MediaPipe) com e sem modalidades auxiliares?
- Q5: Qual o custo computacional incremental por modalidade e o trade-off precisão vs. latência?

## 3. Arquitetura Proposta (Visão Geral)

Pipeline:  
Entrada (por frame / sequência): keypoints 2D (pose + mãos), frame RGB, mapa de profundidade estimado, máscara de pessoa, (opcional) múltiplas vistas.  
1. Pré-processamento: normalização root-centered, escalonamento (opcional), redimensionamento depth/mask, sincronização temporal.  
2. Encoders:
   - Keypoint Encoder (MLP por junta) → tokens (T×J×D).
   - Depth Encoder (CNN leve ou backbone pré‑treinado) → embedding por frame (T×Dd).
   - Segmentation Encoder (CNN leve) → (T×Ds).
   - Video Encoder (3D Conv ou ViT temporal) → (T×Dv) ou (T×Patches×Dv).
   - Multi‑View: empilhamento por câmera e atenção cruzada (futuro).  
3. Fusão Latente: broadcasting de embeddings frame-level somados aos tokens de juntas; Transformer temporal sobre sequência flatten (T×J) ou concat de vistas.  
4. Regressão 3D: cabeça linear → (T×J×3).  
5. (Opcional) Classificador de gesto: opera sobre sequência 3D (ou features latentes agregadas) para F1/Acurácia.

## 4. Espaço Latente Multimodal

Representado por um conjunto de tokens de dimensão unificada D:  
L = { k_{t,j}, d_t, s_t, v_t, (cameras) } → projeções para D e combinação (soma + atenção).  
Critérios de design: (i) modularidade (ativar/desativar modalidades), (ii) baixa acoplagem de pré-processamento, (iii) escalável para multi-view (adicionar dimensão V).  
Futuro: contraste supervisionado (pose 3D como alvo) ou auto-regressão para robustez temporal.

## 5. Metodologia Detalhada

Etapas Incrementais:  
Fase A (Baseline): keypoints → lifting MLP/TCN.  
Fase B: adicionar depth monocular (Depth Anything 2 / Depth Pro) → análise de ganho relativo.  
Fase C: adicionar segmentação (DeepLab / SAM recortado em máscara simples).  
Fase D: incorporar video embeddings (3D CNN simples → substituível por modelo pré‑treinado).  
Fase E: integração multi‑view (atenção entre vistas).  
Fase F: ablações e otimizações (mixed precision, quantização leve se necessário).  

## 6. Plano Experimental

Ablações Principais (cada linha treinada sob mesmo protocolo):
1. 2D somente (baseline classificador de gestos)  
2. Lifting 3D (keypoints somente)  
3. 3D + depth  
4. 3D + depth + seg  
5. 3D + depth + seg + video embeddings  
6. Multi‑view + (todas modalidades) (quando disponível)  
7. Robustez: injeção de ruído nos keypoints vs. latente multimodal.

Métricas:
- Estimação 3D: MPJPE, PA-MPJPE (obrigatórios); (opcional) erro relativo por membro (upper/lower body).  
- Classificação: F1 macro, Acurácia, Latência média (ms/frame).  
- Eficiência: parâmetros, FLOPs aproximados, throughput FPS.

Critérios de Sucesso (indicativos):
- Redução ≥ X% em MPJPE ao passar de keypoints→multimodal (definir X após baseline).  
- Ganho F1 macro ≥ Y% (multimodal vs. 2D puro).  
- Latência pipeline ≤ 60 ms/frame (CPU otimizada ou CPU+GPU leve).

## 7. Datasets e Dados

- Públicos candidatos: 20BN-Jester (gestos manuais), SHREC (gestos de mão), (se disponível) dataset multi‑view com pose 3D anotada (Human3.6M apenas para estudo metodológico – atenção a licenças).  
- Coleta própria: pequeno conjunto custom (webcam) para teste de generalização e ruído real.  
- Armazenamento: usar `.npz` com chaves (keypoints, depth, mask, video_rgb, pose3d).  
- Privacidade: descartar frames brutos quando possível, manter apenas derivados (keypoints/máscara/depth).  

## 8. Cronograma (Macro)

- Semana 1–2: Baseline + integração depth monocular.  
- Semana 3–4: Segmentação + refino do pipeline de treinamento.  
- Semana 5–6: Video embeddings + otimizações (caching, AMP).  
- Semana 7–8: Multi‑view (se houver dataset) + análise comparativa.  
- Semana 9: Ablations completas + coleta própria + robustez a ruído.  
- Semana 10: Escrita de resultados, gráficos, relatório final, demo.  

## 9. Riscos e Mitigações

- Falta de dataset multi‑view acessível: focar em single‑view multimodal e simulação sintética (render pose3D→2D de múltiplas vistas).  
- Custo computacional de video embeddings: iniciar com 3D CNN leve; só depois testar ViT/VideoMAE.  
- Profundidade inconsistente (depth monocular instável): normalizar por mediana temporal ou z-score por sequência.  
- Overfitting em dataset pequeno: regularização (dropout, augment jitter em keypoints, redução de dimensões).  
- Latência acima da meta: profiling + redução de resolução depth/mask + pruning do Transformer.

## 10. Aspectos Éticos e Privacidade

- Minimizar armazenamento de vídeo bruto; preferir derivados anônimos (keypoints, máscaras, depth normalizado).  
- Avisar participantes em coleta própria; evitar identificação facial (não armazenar rosto RGB).  
- Garantir que modelos pré-treinados usados respeitam licenças (Depth Anything 2, SAM, etc.).

## 11. Implementação no Repositório

Estrutura já refletida no código atual: `multimodal_lifter.py`, encoders em `src/features/`, config `multimodal.yaml`, script `train_multimodal.py`, extração `extract_modalities.py`. Próximos incrementos: script de inferência multimodal, integração real de depth/seg, caching de embeddings, suporte multi‑view (dimensão adicional V em tokens) e testes automatizados.

## 12. Referências (seleção)

- LiftPose3D (domain adaptation + lifting).  
- MPL (Multi-view Pose Lifting via Transformer).  
- Depth Anything 2 / Depth Pro (monocular depth).  
- Segment Anything (SAM) / DeepLab v3.  
- VideoMAE / Timesformer (embeddings de vídeo).  
- Surveys em 2D/3D pose estimation e gesture recognition.

## 13. Estado da Arte (Resumo Sintético)

Avanços em lifting 2D→3D e visão multimodal apontam para arquiteturas que integram sinais complementares para reduzir ambiguidades espaciais. Depth monocular fornece proxy de escala/oclusão mesmo sem multi‑view; segmentação estabiliza keypoints filtrando ruído de fundo; embeddings de vídeo capturam dinâmica e contexto global além da geometricidade dos joints; multi‑view (quando disponível) consolida reconstrução precisa. O projeto posiciona-se como ponte entre abordagens minimalistas (apenas keypoints) e pipelines ricos em sinais, analisando custo/benefício de cada modalidade na prática.

