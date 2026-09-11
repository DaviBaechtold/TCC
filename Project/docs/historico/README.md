# Registro histórico — planejamento anterior à pausa

Os documentos desta pasta foram escritos entre outubro e novembro de 2025, antes
da pausa de dez meses do projeto. **Eles não descrevem o sistema atual** e são
mantidos apenas como registro da evolução do trabalho.

O que mudou desde então, e por que eles contradizem o estado corrente:

| Estes documentos dizem | O que é verdade hoje |
|---|---|
| Estágio 1 é YOLOv11-pose | É RTMDet-nano, com 1,01M parâmetros, e é dispensável por configuração |
| Estimador é RTMPose-m em 256×192 | É RTMW-x em 384×288, escolhido por medição comparativa |
| Abordagem bottom-up recomendada | O pipeline é top-down; os números de FPS citados a favor do bottom-up não tinham condição de medição declarada |
| Meta de whole-body AP > 75% | Acima do estado da arte mundial (70,2%). Meta atual: 70% |
| Lifting 3D restrito a 17 juntas | É full-body, 133 keypoints, treinado no H3WB |
| Checkpoint do projeto em `work_dirs/test_minimal5/` | Descartado. Mede 0,4373 de AP contra 0,6930 do atual |

A documentação corrente do repositório é o `CLAUDE.md` na raiz, e a
especificação do sistema é o Projeto Físico, no repositório de documentos.
