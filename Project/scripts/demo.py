#!/usr/bin/env python
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import sys
import os
import warnings

# Ensure 'Project/src' is importable when running this file directly
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mediapipe_extractor import MediaPipeHandExtractor, ExtractConfig
from src.models.transformer import TransformerClassifier
from src.utils.train_utils import load_checkpoint


def preprocess_seq(frames: list, target_len: int) -> torch.Tensor:
    # frames list of [H,K,2] or [K,2], flatten per frame
    if len(frames) == 0:
        return torch.zeros(1, target_len, 21 * 2)
    seq = []
    for f in frames:
        if f.ndim == 3:
            f = f.reshape(-1, 2)
        seq.append(f.reshape(-1))
    seq = np.array(seq, dtype=np.float32)
    # simple pad/trim
    T, F = seq.shape
    out = np.zeros((target_len, F), dtype=np.float32)
    if T >= target_len:
        out[:] = seq[-target_len:]
    else:
        out[-T:] = seq
    # normalize
    out = (out - out.mean(0, keepdims=True)) / (out.std(0, keepdims=True) + 1e-6)
    return torch.from_numpy(out).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ckpt_path}. Verifique o caminho absoluto ou rode com --checkpoint '/media/davs/SSD/TCC - Database/processed/runs/transformer/best.pt'")

    # Safe device selection: fall back to CPU if CUDA fails at runtime
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            # Force a tiny CUDA kernel to ensure runtime compatibility
            t = torch.ones(1, device='cuda')
            _ = t.sin_()
            device = 'cuda'
        except Exception as e:
            warnings.warn(f"CUDA indisponível/inesperada falha em kernel, usando CPU. Motivo: {e}")
    extractor = MediaPipeHandExtractor(ExtractConfig(hands=1, static_image_mode=False))

    # Determine num_classes from checkpoint metadata to avoid size-mismatch
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    extra = ckpt.get('extra', {})
    class_to_idx = extra.get('class_to_idx')
    if class_to_idx is not None:
        num_classes = len(class_to_idx)
    else:
        # Fallback: infer from classifier head shape
        head_w = ckpt['model'].get('head.3.weight')
        if head_w is None:
            raise SystemExit("Não foi possível inferir num_classes do checkpoint. Treine novamente ou forneça metadata.")
        num_classes = int(head_w.shape[0])

    # Build model with correct output dimension and load weights
    model = TransformerClassifier(input_dim=42, num_classes=num_classes)
    model.load_state_dict(ckpt['model'])
    idx_to_class = None
    if class_to_idx:
        # invert mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    model.to(device).eval()

    cap = cv2.VideoCapture(0)
    frames = []
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seq = extractor.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            coords = np.full((1, 21, 2), np.nan, dtype=np.float32)
            if seq.multi_hand_landmarks:
                h, w, _ = frame.shape
                lm = seq.multi_hand_landmarks[0]
                for j, p in enumerate(lm.landmark[:21]):
                    coords[0, j, 0] = p.x * w
                    coords[0, j, 1] = p.y * h
            frames.append(coords)
            if len(frames) > args.seq_len:
                frames = frames[-args.seq_len:]

            x = preprocess_seq(frames, args.seq_len).to(device)
            m = torch.ones(1, x.shape[1], device=device)
            logits = model(x, m)
            pred = int(logits.argmax(dim=1).item())
            label = idx_to_class.get(pred, str(pred)) if idx_to_class else str(pred)
            cv2.putText(frame, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
