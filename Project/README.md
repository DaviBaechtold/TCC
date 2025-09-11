# Temporal Transformer-Based Gesture Recognition (2D Skeletons)

This project recognizes dynamic hand/body gestures from monocular RGB videos. It extracts 2D keypoints with MediaPipe, builds temporal sequences, and classifies them using a Transformer encoder. Includes a baseline (LSTM/MLP), training loop, evaluation, and a simple real-time demo.

## Layout
- `src/data/`: data extraction and datasets
- `src/models/`: transformer, baselines
- `src/utils/`: training, metrics, viz
- `scripts/`: CLI entry-points

## Quickstart
1) Create a Python env and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Project/requirements.txt
```
2) Prepare data

Option A — From standalone videos (mp4/avi): extract keypoints into `.npz` sequences
```bash
python Project/scripts/extract_keypoints.py --videos /path/to/videos --out Project/data/npz --max-frames 120
```
Option B — From 20BN-Jester frames on SSD: convert frame folders to `.npz` and a manifest

- Expected dataset root (example):
	- `/media/davs/SSD/TCC - Database/20bn-jester/Train/<video_id>/*.jpg`
	- `/media/davs/SSD/TCC - Database/20bn-jester/Validation/<video_id>/*.jpg`
- Run conversion per split to SSD:
```bash
python Project/scripts/build_manifest_from_jester.py \
	--root "/media/davs/SSD/TCC - Database/20bn-jester" \
	--split Train \
	--out "/media/davs/SSD/TCC - Database/processed/npz_jester" \
	--manifest "/media/davs/SSD/TCC - Database/processed/manifest_train.csv" \
	--stride 2 --max-frames 64

python Project/scripts/build_manifest_from_jester.py \
	--root "/media/davs/SSD/TCC - Database/20bn-jester" \
	--split Validation \
	--out "/media/davs/SSD/TCC - Database/processed/npz_jester" \
	--manifest "/media/davs/SSD/TCC - Database/processed/manifest_val.csv" \
	--stride 2 --max-frames 64
```
Optionally merge manifests:
```bash
python Project/scripts/merge_manifests.py \
	--inputs "/media/davs/SSD/TCC - Database/processed/manifest_train.csv" \
					 "/media/davs/SSD/TCC - Database/processed/manifest_val.csv" \
	--out "/media/davs/SSD/TCC - Database/processed/manifest_full.csv" --dedup
```

3) Configure training

Edit `Project/configs/transformer.yaml` to point `data.manifest` and `work_dir` to the SSD (examples already set):

- `data.manifest: "/media/davs/SSD/TCC - Database/processed/manifest_train.csv"`
- `work_dir: "/media/davs/SSD/TCC - Database/processed/runs/transformer"`

4) Train
```bash
python Project/scripts/train.py --config Project/configs/transformer.yaml
```

Resume training later (after Ctrl+C):
```bash
# resume from rolling last checkpoint
python Project/scripts/train.py --config Project/configs/transformer.yaml --resume last

# or from best checkpoint
python Project/scripts/train.py --config Project/configs/transformer.yaml --resume best
```

5) Evaluate
```bash
# Evaluate a trained checkpoint on a manifest
python Project/scripts/evaluate.py \
	--manifest "/media/davs/SSD/TCC - Database/processed/manifest_val.csv" \
	--checkpoint "/media/davs/SSD/TCC - Database/processed/runs/transformer/best.pt" \
	--seq-len 64
```

6) Quick camera test (show face/hand/pose landmarks and bounding boxes)
```bash
python Project/scripts/camera_test.py --mirror
```

7) Real-time classification demo (requires a trained checkpoint)
```bash
python Project/scripts/demo.py --checkpoint runs/transformer/best.pt
```

See `Project/configs/` for hyperparameters.

## Solução de Problemas (Troubleshooting)

- CUDA/GPU não usada ou erro de arquitetura (ex.: sm_61):
  - O script faz fallback automático para CPU.
  - Para forçar CPU: adicione `device: cpu` no YAML ou rode com `CUDA_VISIBLE_DEVICES=""`.
  - Para usar GPU, instale o PyTorch compatível com sua GPU/CUDA.

- ModuleNotFoundError: 'src' ou imports quebrados:
  - Execute os scripts a partir da raiz do repositório: `/home/davs/Documents/TCC`.
  - Opcional: exporte `PYTHONPATH` uma vez: `export PYTHONPATH="$PWD/Project:$PYTHONPATH"`.

- Caminhos no SSD com espaço no nome:
  - Sempre use aspas: "/media/davs/SSD/TCC - Database/...".
  - Em YAML, mantenha os caminhos entre aspas.

- Manifesto (.csv) não encontrado, vazio ou inconsistente:
  - Cabeçalho deve ser `path,label` e cada linha apontar para um `.npz` existente.
  - Se houver aviso de `class_to_idx` ao retomar, use o mesmo manifest do treino original.

- Extração (MediaPipe/OpenCV) falhando:
  - Em Linux, instale libs do sistema para OpenCV GUI/GL:
    - `sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0`
  - Avisos de delegates do MediaPipe/TFLite podem ser ignorados; para estabilidade, execute em CPU.
  - Lento/pesado? Use `--stride 2`, `--max-frames 64` e `--limit` para validar.

- Permissão negada ao salvar no SSD:
  - Verifique se a unidade está montada com escrita e você tem permissão no diretório `processed/`.
  - Ajuste permissões (ex.): `sudo chown -R "$USER":"$USER" "/media/davs/SSD/TCC - Database/processed"`.

- Retomar treinamento interrompido:
  - `--resume last` (checkpoint rolante) ou `--resume best`.
  - Não altere classes/labels entre sessões; se mudar, recomece do zero.

- Desempenho e memória:
  - Em CPU, use `num_workers: 2-4` e reduza `batch_size`/`seq_len` se faltar RAM.
  - `pin_memory` é desativado automaticamente em CPU.

- Erros de YAML:
  - Coloque caminhos entre aspas e use espaços (sem tabs).

- Diagnóstico rápido do ambiente:
  - `which python && python -V`
  - `pip show torch mediapipe opencv-python`
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

Se o problema persistir, compartilhe o comando usado, a mensagem completa e as últimas ~20 linhas do log.
