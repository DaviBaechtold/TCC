# Lifting 2D→3D para Reconhecimento de Gestos (TCC)

Projeto minimalista para estudo de lifting de poses 2D→3D aplicado a reconhecimento de gestos.

Documento da proposta: `Doc/Project Propose/Proposta.md`.

## Estrutura
- `Project/src/` — código-fonte (placeholder)
- `Project/configs/` — configurações (placeholder)
- `Project/data/` — dados (placeholder)
- `Project/scripts/camera_test.py` — teste de webcam (MediaPipe Holistic)

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

## Roadmap de Implementação
1) `configs/lifter.yaml`: hiperparâmetros do lifter 2D→3D (MPJPE/PA-MPJPE)
2) `src/models/lifter.py`: MLP/TCN de lifting e utilitários de normalização
3) `scripts/train_lifter.py`: treino do lifter em dados sintéticos ou pares 2D↔3D
4) `scripts/lift_sequences.py`: conversão de sequências 2D `.npz` → 3D `.npz`
5) `scripts/evaluate_3d_metrics.py`: MPJPE, PA-MPJPE (e MPVE se aplicável)
6) (Opcional) `scripts/depth_segment_features.py`: extração de sinais auxiliares

Consulte o documento da proposta para o protocolo experimental (P1–P5), métricas e critérios de sucesso.

## Comandos rápidos (modo sintético)
Treinar o lifter (gera checkpoint em `Project/data/lifter_runs/lifter_best.pt`):
```bash
python Project/scripts/train_lifter.py --config Project/configs/lifter.yaml --synthetic --device cpu
```

Avaliá-lo:
```bash
python Project/scripts/evaluate_3d_metrics.py \
	--checkpoint Project/data/lifter_runs/lifter_best.pt \
	--num-joints 17 --n-val 1000 --device cpu
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
