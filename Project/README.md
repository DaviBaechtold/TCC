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
python Project/scripts/camera_test.py --camera 1 --width 1280 --height 720 --fps 30 --mirror
```

## Roadmap de Implementação
1) `configs/lifter.yaml`: hiperparâmetros do lifter 2D→3D (MPJPE/PA-MPJPE)
2) `src/models/lifter.py`: MLP/TCN de lifting e utilitários de normalização
3) `scripts/train_lifter.py`: treino do lifter em dados sintéticos ou pares 2D↔3D
4) `scripts/lift_sequences.py`: conversão de sequências 2D `.npz` → 3D `.npz`
5) `scripts/evaluate_3d_metrics.py`: MPJPE, PA-MPJPE (e MPVE se aplicável)
6) (Opcional) `scripts/depth_segment_features.py`: extração de sinais auxiliares

Consulte o documento da proposta para o protocolo experimental (P1–P5), métricas e critérios de sucesso.
