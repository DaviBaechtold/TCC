#!/usr/bin/env python
import argparse
from pathlib import Path
import csv
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Build manifest CSV from a directory of .npz sequences.")
    ap.add_argument('--npz-dir', required=True, help='Directory containing .npz files')
    ap.add_argument('--out', required=True, help='Output manifest CSV path')
    ap.add_argument('--pattern', default='*.npz', help='Glob pattern for files (default: *.npz)')
    ap.add_argument('--limit', type=int, default=None, help='Optional limit for quick tests')
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    files = sorted(npz_dir.rglob(args.pattern))
    if args.limit:
        files = files[: args.limit]
    rows = []
    for p in files:
        try:
            d = np.load(p, allow_pickle=True)
            label = d['label'].item() if 'label' in d else 'unknown'
            rows.append((str(p), str(label)))
        except Exception as e:
            print(f"WARN: failed to read {p}: {e}")
            continue

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'])
        w.writerows(rows)
    print(f"Wrote manifest: {args.out} with {len(rows)} entries")


if __name__ == '__main__':
    main()
