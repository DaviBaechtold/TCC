1. Visão Geral

Este estudo de caso aprofunda a motivação, tecnologias e escolhas de arquitetura para gerar um espaço latente multimodal integrando: (i) keypoints 2D (MediaPipe), (ii) depth monocular (Depth Anything 2 ou Depth Pro), (iii) segmentação humana, (iv) embeddings de vídeo temporais, e (v) extensões multi‑view. O objetivo é melhorar robustez geométrica, temporal e semântica do lifting 2D→3D e, em consequência, elevar a qualidade na classificação de gestos.

2. Motivação Multimodal

Limitações de somente keypoints 2D: ambiguidade de profundidade, sensibilidade a oclusões, perda de contexto (velocidade, dinâmica global). A fusão multimodal adiciona:

Depth: proxy de escala / ordem relativa (quem está à frente / atrás) mesmo sem múltiplas câmeras.

Segmentação: estabiliza detecção e reduz ruído de fundo (melhora consistência espacial dos keypoints).

Video Embeddings: capturam padrões temporais (aceleração, ritmo) e textura (mãos parcialmente detectadas).

Multi‑View: reduz ambiguidades estruturais, melhora triangulação implícita, reforça correspondência inter-vistas.

3. Monocular Depth como Proxy

3.1. Modelos Recomendados

Depth Anything 2: foco em generalização ampla, bom equilíbrio velocidade/qualidade.

Depth Pro: arquitetura otimizada para performance; checar licença e suporte.

3.2. Integração no Pipeline

Inferir mapa de depth por frame (RGB→Depth).

Normalizar (ex.: min-max local por sequência ou z-score).

Extrair features: CNN leve → embedding por frame (Dd).

Broadcast + soma aos tokens de juntas ou concatenar em atenção multi-token.

3.3. Estratégias de Feature Engineering

Estratégia Vantagem Custo Média global + desvio Barato Perde estrutura espacial Patches alinhados a bounding box corporal Preserva regiões Moderado CNN + AdaptiveAvgPool Captura padrões médias + robusto Moderado ViT pequeno sobre depth Maior capacidade Alto

3.4. Normalizações Possíveis

Depth relativo: d' = (d - median(seq)) / MAD.

Clipping percentílico (ex.: [2%,98%]) para mitigar outliers.

Máscara humana aplicada antes da média para remover fundo distante.

4. Segmentação Humana

Modelos: DeepLab v3, Mask2Former, SAM (Segment Anything) para anotação offline. Para tempo real, usar rede leve (MobileNet backbone). Máscara binária final reduz ruído de background, permitindo:

Filtragem de keypoints instáveis fora da silhueta.

Ativação de pesos de confiança (keypoint score * presença na máscara).

Foco de depth nas regiões relevantes (mask * depth).

5. Embeddings de Vídeo

5.1. Opções

3D CNN leve (e.g., R(2+1)D / C3D simplificado) – rápido enquanto prototipa.

VideoMAE, TimeSformer, X3D, SlowFast (pré‑treinados em Kinetics) – transferência de dinâmica complexa.

5.2. Extração

Janela deslizante (T_clip) com sobreposição (stride < T_clip) → média ou atenção temporal.

Congelar pesos pré‑treinados inicialmente; só treinar projeções.

5.3. Redução de Dimensão

PCA offline ou linear bottleneck (Linear Dv→D).

Ativação GELU + LayerNorm antes de fusão.

6. Multi‑View

Quando múltiplas câmeras são acessíveis:

Sincronização: timestamps ou alinhamento por frame index.

Fusão: (i) concatenação de tokens por vista e atenção global; (ii) atenção hierárquica (intra‑view → inter‑view); (iii) pooling espacial 2D→token por vista + atenção cruzada.

Regularização: consistência reprojetada (pred 3D → projeção 2D deve aproximar keypoints originais por vista).

Dados sintéticos (AMASS + SMPL): gerar cenas multi‑view (vide pipeline MPL) para pré‑treino.

7. Espaço Latente Unificado

Tokens:

k_{t,j}: junta j no tempo t.

d_t: embedding depth.

s_t: embedding segmentação.

v_t: embedding vídeo (ou múltiplos patches).

(opcional) c_{t,v}: token agregado por câmera.

Fusão: ( z = \text{Transformer}([k] + f(d_t,s_t,v_t,c_{t,v})) ) com broadcasting ou cross‑attention.

Perdas auxiliares possíveis:

Reconstrução de depth reduzido (autoencoder parcial).

Consistência temporal (|k_{t+1}-k_t| regularizado vs. velocidade derivada de vídeo).

Consistência multi‑view (projeção inversa).

8. Métricas

Categoria Métricas Observações Lifting 3D MPJPE, PA-MPJPE Avaliar por região (tronco, membros) Robustez MPJPE sob ruído sintético Injetar jitter gaussiano em 2D Classificação F1 macro, Acurácia, Confusion Matrix Avaliar cada ablação Eficiência Latência (ms/frame), FPS, FLOPs estimados Separar CPU vs. GPU Ablation Gain ΔMPJPE, ΔF1 vs. baseline Documentar incremental

9. Datasets Relevantes

Nome Tipo Uso Observações 20BN-Jester RGB gestos mão/cotovelo Classificação + lifting sintético Não possui GT 3D SHREC (Hand Gestures) Mão, sensor & RGB Gestos de mão Útil para generalização Human3.6M Multi-view + 3D GT Lifting supervisionado / multi-view Licença restrita CMU Panoptic Multi-view + 3D Pré-treino / consistência espacial Grande e pesado AMASS MoCap agregado Síntese 2D multi-view Para gerar pares 2D–3D MHP (Mesh-based Human Pose Generator) Sintético Render multi-view, depth, máscara Facilita supervisão completa NTU RGB+D Ações multi-modal (RGB+D+S) Transfer learning de dinâmica Prever adaptação de embeddings EgoSign / WLASL (língua de sinais) Sequências de sinais Testar transferência de embeddings gesto Pode requerer mapeamento de joints

10. Bases para Depth e Segmentação

Depth Anything 2 (GitHub: https://github.com/LiheYoung/Depth-Anything) – checar v2 quando público.

MiDaS (Intel ISL) – alternativa consolidada.

Segment Anything (SAM) – geração de máscaras offline (https://github.com/facebookresearch/segment-anything).

DeepLab v3 (torchvision) – inferência rápida.

11. Repositórios de Referência (GitHub)

MMPose (OpenMMLab): https://github.com/open-mmlab/mmpose

Detectron2 (infra segmentação): https://github.com/facebookresearch/detectron2

Segment Anything: https://github.com/facebookresearch/segment-anything

VideoMAE: https://github.com/MCG-NJU/VideoMAE

TimeSformer: https://github.com/facebookresearch/TimeSformer

SlowFast: https://github.com/facebookresearch/slowfast

X3D (via PyTorchVideo): https://github.com/facebookresearch/pytorchvideo

LiftPose3D: (paper / reimplementações diversas)

MPL Multi-view Lifter: (ver paper na pasta de referências)

12. Exemplos de Pipelines / Inspiração

Objetivo Recurso Pose 2D tempo real MediaPipe Holistic Lifting baseline MLP/TCN (repo atual lifter.py) Depth monocular Depth Anything / MiDaS wrapper Segmentation Deeplab (torchvision.models.segmentation) Video embeddings VideoMAE fine-tune congelando backbone Multi-view sintético Render SMPL (AMASS) + projeção para K vistas

13. Estratégia de Treinamento

Pré‑treino (sintético AMASS→2D+ruído) para inicializar regressor 3D.

Ajuste em dataset real (Human3.6M ou subset similar) se licença permitir.

Adição incremental de depth e segmentação (congelar pesos iniciais).

Introdução de embeddings de vídeo (freezing + fine‑tune leve da projeção).

Multi‑view: atenção cruzada + regularização de reprojeção.

Fine‑tune final para classificação de gestos (cabeça separada ou pooling sobre 3D).

14. Considerações sobre Kinetic vs. Infravermelho

Kinetic (RGB + inferred 3D): depende de iluminação; mais versátil e barato (câmeras comuns).

Infravermelho / Profundidade ativa (Kinect-like): fornece profundidade direta mas sofre com alcance limitado, ruído em superfícies brilhantes.

O projeto assume cenário de custo reduzido (RGB + depth monocular). Multi‑view mitiga parte da perda de fidelidade em comparação a sensores dedicados.

15. Sinais para Língua de Sinais e Gestos Finos

Bases (WLASL, EgoSign) usam keypoints detalhados de mãos. Integração de embeddings de vídeo ajuda quando detecção de dedos é parcial.

Transfer learning: congelar backbone de vídeo pré‑treinado em Kinetics e adaptar projeção para gestos específicos.

16. Roadmap Técnico Resumido

Fase Entrega Modalidades A Baseline 3D lifter keypoints B + Depth keypoints + depth C + Segmentação + mask D + Vídeo Embeddings + vídeo E + Multi‑View + vistas F Ablations & Otimizações todos

17. YouTube – Material de Apoio

Tema Link Pose Estimation Overview https://www.youtube.com/watch?v=pW6nZXeWlGM MediaPipe Holistic https://www.youtube.com/watch?v=qV6e4l5JHJ8 Monocular Depth (MiDaS) https://www.youtube.com/watch?v=2lprC0yYeFw Segment Anything Explicação https://www.youtube.com/watch?v=Jp0o8b0wJ6k VideoMAE Intro https://www.youtube.com/watch?v=pFfCdf0JSpA SlowFast Networks https://www.youtube.com/watch?v=YRhxdVk_sIs Multi-view Pose (ex. Panoptic) https://www.youtube.com/watch?v=0hU6qQw1CwU Transformer for Vision https://www.youtube.com/watch?v=TrdevFK_am4

18. Referências Bibliográficas (Seleção)

(Organizar futuramente em BibTeX; aqui em formato livre.)

Pavllo et al. (LiftPose3D) – Lifting 2D human pose to 3D with temporal modeling and domain adaptation.

MPL: Multi-view Pose Lifting using Transformers (paper na pasta de referências).

Survey: An In-Depth Analysis of 2D and 3D Pose Estimation Techniques (PDF na pasta).

Depth Anything / Depth Pro – repositórios e whitepapers.

VideoMAE: Masked Autoencoders are Data-Efficient Learners for Video Understanding.

TimeSformer: Is Space-Time Attention All You Need for Video Understanding?

SlowFast Networks for Video Recognition.

Segment Anything Model (SAM) – meta AI.

DeepLab v3 – Chen et al.

AMASS: Archive of Motion Capture as Surface Shapes.

Human3.6M Dataset paper.

CMU Panoptic Studio.

WLASL / EgoSign (língua de sinais) – gesture/sign benchmarks.

Kinetics dataset (pré-treinamento de vídeo).