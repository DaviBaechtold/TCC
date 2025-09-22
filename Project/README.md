# Lifting 2D→3D Multimodal para Reconhecimento de Gestos (TCC)

Projeto evoluído: de um lifter minimalista 2D→3D para uma arquitetura multimodal que integra
`keypoints` + `depth (monocular)` + `human segmentation` + `video embeddings` em um espaço latente
temporal para melhorar robustez e semântica de gestos. Multi‑view futuro pode substituir depth
monocular quando disponível.

Documento da proposta: `Doc/Project Propose/Proposta.md`.

## Estrutura (principal)
- `Project/configs/lifter.yaml` — lifting simples (baseline)
- `Project/configs/multimodal.yaml` — configuração multimodal (seq temporal, fusão)
- `Project/src/models/lifter.py` — MLP / TCN baseline
- `Project/src/models/multimodal_lifter.py` — modelo multimodal (Transformer de fusão)
- `Project/src/features/` — encoders de depth, máscara e vídeo (stubs leves)
- `Project/scripts/train_lifter.py` — treino baseline
- `Project/scripts/train_multimodal.py` — treino multimodal (sintético ou dataset NPZ)
- `Project/scripts/extract_modalities.py` — extração rápida de keypoints + pseudo depth/mask
- `Project/scripts/lift_sequences.py` — inferência (baseline)
- `Project/scripts/camera_test.py` — teste de câmera / keypoints

## Requisitos
Crie e ative o ambiente Python e instale dependências:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Project/requirements.txt
```

Se necessário em Linux para OpenCV GUI:
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

## Teste rápido com a Logitech C922
```bash
# The camera numeric index can vary between machines. First list devices:
python Project/scripts/camera_test.py --list-devices

# Then run using the working index (e.g. 2) or, more reliably, the device path:
python Project/scripts/camera_test.py --camera 2 --width 1280 --height 720 --fps 30 --mirror
python Project/scripts/camera_test.py --camera /dev/video2 --width 1280 --height 720 --fps 30 --mirror
```

## Arquitetura Multimodal (Resumo)
1. `Keypoints Encoder`: MLP por junta gera tokens (B,T,J,D)
2. `Depth Encoder`: CNN leve em mapas de profundidade monocular → (B,T,Dd)
3. `Segmentation Encoder`: CNN leve em máscara binária de pessoa → (B,T,Ds)
4. `Video Encoder`: 3D CNN simples → (B,T,Dv)
5. `Fusion Transformer`: Soma tokens de keypoints com embeddings broadcast de frame (depth+mask+vídeo) e aplica Transformer temporal sobre sequência flatten (T×J)
6. `Head`: Linear para (x,y,z) por junta → (B,T,J,3)

Cada modalidade pode ser desligada em `configs/multimodal.yaml`. Depth real (Depth Anything 2 / Depth Pro) e segmentação avançada podem ser plugadas substituindo os stubs.

## Formato de Dataset NPZ Multimodal
Chaves esperadas (as ausentes são ignoradas):
```
keypoints: (N,T,J,2)
pose3d:    (N,T,J,3)   # ground truth
depth:     (N,T,Hd,Wd)
mask:      (N,T,Hm,Wm)
video_rgb: (N,T,3,Hv,Wv)
```

## Fluxo de Uso Multimodal
1. Extrair modalidades de um vídeo cru:
```bash
python Project/scripts/extract_modalities.py \
	--video input.mp4 --out Project/data/sample_multi.npz --max-frames 128 --stride 1
```
2. (Opcional) Substituir depth/mask pseudo por versões reais.
3. Treinar (sintético multimodal de demonstração):
```bash
python Project/scripts/train_multimodal.py --config Project/configs/multimodal.yaml --synthetic --device cpu
```
4. Treinar em dataset real (supondo `pose3d` disponível):
```bash
python Project/scripts/train_multimodal.py \
	--config Project/configs/multimodal.yaml \
	--dataset Project/data/sample_multi.npz --device cuda:0
```

## Baseline Original (Simples)
Mantido para comparações (MPJPE / PA-MPJPE) conforme roadmap inicial.

## Comandos rápidos (modo sintético)
Treinar o lifter (gera checkpoint em `Project/data/lifter_runs/lifter_best.pt`):
```bash
python Project/scripts/train_lifter.py --config Project/configs/lifter.yaml --synthetic --device cpu
```

Converter um `.npz` 2D para 3D:
```bash
python Project/scripts/lift_sequences.py \
	--input /path/to/input_2d.npz \
	--output /path/to/output_3d.npz \
	--checkpoint Project/data/lifter_runs/lifter_best.pt \
	--root-center --device cpu
```

### Treino temporal (TCN)
Edite `Project/configs/lifter.yaml` e ajuste:
```yaml
model:
	type: tcn
seq_len: 16
```
Rode:
```bash
python Project/scripts/train_lifter.py --config Project/configs/lifter.yaml --synthetic --device cpu --n-train 4000 --n-val 800
```

### Captura de keypoints reais (webcam)
```bash
# Prefer listing devices first and using a device path if index is unreliable:
python Project/scripts/capture_keypoints.py --out Project/data/captura_seq.npz --camera /dev/video2 --frames 300 --mirror --show
```
Depois fazer lifting:
```bash
python Project/scripts/lift_sequences.py \
	--input Project/data/captura_seq.npz \
	--output Project/data/captura_seq_3d.npz \
	--checkpoint Project/data/lifter_runs/lifter_best.pt --root-center
```

## Próximos Passos Sugeridos
- Integrar Depth Anything 2 (converter para inferência Torch, normalizar 0–1).
- Substituir máscara pseudo por Segment Anything / DeepLab v3.
- Incorporar embeddings CLIP ou VideoMAE pré-treinados (substituir `Simple3DConvEncoder`).
- Suporte multi-view: agregar features de câmeras distintas antes da fusão (concat + proj). 
- Métricas adicionais: P-MPJPE, velocidade média por junta (dinâmica), robustez a oclusões.

## Notas de Performance
Implementação atual prioriza clareza: não há caching de embeddings nem data pipeline otimizado. Para produção, usar DataLoader com prefetch, mixed precision (`torch.cuda.amp`) e acumulação de gradiente para lotes grandes.

