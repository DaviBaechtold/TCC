#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DataConfig, SkeletonSequenceDataset
from src.models.transformer import TransformerClassifier
from src.utils.train_utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seq-len', type=int, default=64)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset to know classes and input dims
    data_cfg = DataConfig(manifest=args.manifest, seq_len=args.seq_len)
    ds = SkeletonSequenceDataset(data_cfg)
    input_dim = ds[0]['x'].shape[-1]
    num_classes = ds.num_classes

    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    load_checkpoint(args.checkpoint, model)
    model.to(device).eval()

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    ce = torch.nn.CrossEntropyLoss()
    losses = []
    all_y, all_p = [], []
    with torch.no_grad():
        for b in loader:
            x, m, y = b['x'].to(device), b['mask'].to(device), b['y'].to(device)
            logits = model(x, m)
            loss = ce(logits, y)
            losses.append(loss.item())
            all_y.append(y)
            all_p.append(logits.argmax(dim=1))
    y = torch.cat(all_y)
    p = torch.cat(all_p)
    acc = (y == p).float().mean().item()
    print({'loss': sum(losses)/max(1,len(losses)), 'acc': acc})


if __name__ == '__main__':
    main()
