from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    seq = data["sequence"].astype(np.float32)
    # shape [T, H, K, 2] or [T, K, 2] -> make [T, -1]
    if seq.ndim == 4:
        T, H, K, D = seq.shape
        seq = seq.reshape(T, H * K * D)
    elif seq.ndim == 3:
        T, K, D = seq.shape
        seq = seq.reshape(T, K * D)
    else:
        # already flattened per frame
        pass
    # replace NaNs with zeros
    seq = np.nan_to_num(seq)
    return seq


def pad_or_sample(seq: np.ndarray, target_len: int, allow_trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return sequence of length target_len and a boolean mask (1 for valid).

    If seq is shorter, pad with zeros. If longer and allow_trim, center-crop; else
    uniformly sample target_len indices across the sequence.
    """
    T, F = seq.shape
    if T == target_len:
        mask = np.ones((T,), dtype=bool)
        return seq, mask
    if T < target_len:
        out = np.zeros((target_len, F), dtype=seq.dtype)
        out[:T] = seq
        mask = np.zeros((target_len,), dtype=bool)
        mask[:T] = True
        return out, mask
    # T > target_len
    if allow_trim:
        start = (T - target_len) // 2
        out = seq[start : start + target_len]
        mask = np.ones((target_len,), dtype=bool)
        return out, mask
    # uniform sampling
    idx = np.linspace(0, T - 1, target_len).astype(int)
    out = seq[idx]
    mask = np.ones((target_len,), dtype=bool)
    return out, mask


@dataclass
class DataConfig:
    manifest: str
    seq_len: int
    normalize: bool = True
    noise_std: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_len: int = 0


class SkeletonSequenceDataset(Dataset):
    """Loads sequences from a manifest CSV with columns: path,label.

    Each item returns a dict with keys: x [T,F], mask [T], y [int].
    """

    def __init__(self, cfg: DataConfig, class_to_idx: Optional[Dict[str, int]] = None):
        self.cfg = cfg
        df = pd.read_csv(cfg.manifest)
        if not {"path", "label"}.issubset(df.columns):
            raise ValueError("Manifest CSV must have columns: path,label")
        self.paths: List[str] = df["path"].tolist()
        self.labels: List[str] = df["label"].astype(str).tolist()

        if class_to_idx is None:
            classes = sorted(set(self.labels))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.paths)

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return seq
        # Standardize per sample (robust to scale/translation)
        mean = np.mean(seq, axis=0, keepdims=True)
        std = np.std(seq, axis=0, keepdims=True) + 1e-6
        return (seq - mean) / std

    def _augment(self, seq: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if self.cfg.noise_std > 0:
            seq = seq + rng.normal(0, self.cfg.noise_std, size=seq.shape).astype(seq.dtype)
        # time masking
        if self.cfg.time_mask_prob > 0 and self.cfg.time_mask_len > 0:
            T = seq.shape[0]
            if rng.rand() < self.cfg.time_mask_prob:
                start = rng.randint(0, max(1, T - self.cfg.time_mask_len))
                end = min(T, start + self.cfg.time_mask_len)
                seq[start:end] = 0
        return seq

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y_name = self.labels[idx]
        y = self.class_to_idx[y_name]
        seq = _load_npz(path)
        seq = self._normalize(seq)

        seq, mask = pad_or_sample(seq, self.cfg.seq_len)
        # augmentation only on valid region
        rng = np.random.RandomState(abs(hash((idx, len(seq)))) % (2**32))
        valid_len = mask.sum()
        if valid_len > 0:
            aug = self._augment(seq[:valid_len].copy(), rng)
            seq[:valid_len] = aug

        x = torch.from_numpy(seq).float()
        m = torch.from_numpy(mask.astype(np.float32))  # float mask 1.0/0.0
        y = torch.tensor(y, dtype=torch.long)
        return {"x": x, "mask": m, "y": y}


def build_splits(manifest: str, val_split: float, test_split: float, seed: int = 42) -> Tuple[str, str, str]:
    """Create three CSV manifests side-by-side and return their paths.
    Splits are stratified by label.
    """
    df = pd.read_csv(manifest)
    labels = df["label"].astype(str)
    rng = np.random.RandomState(seed)
    # stratified index split
    groups: Dict[str, List[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for gidx in groups.values():
        gidx = gidx.copy()
        rng.shuffle(gidx)
        n = len(gidx)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test_idx.extend(gidx[:n_test])
        val_idx.extend(gidx[n_test : n_test + n_val])
        train_idx.extend(gidx[n_test + n_val :])
    def _write(indices: List[int], name: str) -> str:
        out = df.iloc[indices].reset_index(drop=True)
        out_path = str(Path(manifest).with_name(Path(manifest).stem + f"_{name}.csv"))
        out.to_csv(out_path, index=False)
        return out_path
    return _write(train_idx, "train"), _write(val_idx, "val"), _write(test_idx, "test")
