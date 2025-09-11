#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
import os
import csv

# Ensure 'Project' root is on sys.path to import src package
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mediapipe_extractor import MediaPipeHandExtractor, ExtractConfig, save_npz


def collect_frame_paths(seq_dir: Path):
    # Jester frames are numbered 1..N as JPEGs inside folder named by video_id
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(sorted(seq_dir.glob(ext)))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to 20bn-jester root (contains Train/Validation/Test and CSVs)')
    ap.add_argument('--split', required=True, choices=['Train', 'Validation', 'Test'])
    ap.add_argument('--csv', help='Path to split CSV (if omitted, uses <root>/<split>.csv)')
    ap.add_argument('--out', required=True, help='Output folder for .npz sequences')
    ap.add_argument('--manifest', required=True, help='Path to write manifest CSV with columns path,label')
    ap.add_argument('--hands', type=int, default=1)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='Limit number of sequences to process (for quick tests)')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else root / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load rows: expect header with 'video_id','label', ...
    rows = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        # map columns
        try:
            vid_idx = header.index('video_id')
            label_idx = header.index('label')
        except ValueError:
            raise RuntimeError("CSV must contain columns 'video_id' and 'label'")
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if not parts or len(parts) <= max(vid_idx, label_idx):
                continue
            rows.append((parts[vid_idx], parts[label_idx]))

    extractor = MediaPipeHandExtractor(ExtractConfig(max_frames=args.max_frames, stride=args.stride, hands=args.hands))

    manifest_rows = []
    processed = 0
    for video_id, label in rows:
        seq_dir = root / args.split / video_id
        if not seq_dir.exists():
            print(f"WARN: missing frames folder {seq_dir}, skipping")
            continue
        imgs = collect_frame_paths(seq_dir)
        if not imgs:
            print(f"WARN: no images in {seq_dir}, skipping")
            continue
        sequence = extractor.extract_image_sequence([str(p) for p in imgs])
        out_path = out_dir / f"{args.split.lower()}_{video_id}.npz"
        save_npz(sequence, str(out_path), label=label)
        manifest_rows.append((str(out_path), label))
        print(f"Saved {out_path} T={sequence.shape[0]} label={label}")
        processed += 1
        if args.limit and processed >= args.limit:
            break

    # Write manifest
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'])
        w.writerows(manifest_rows)
    print(f"Wrote manifest: {args.manifest} with {len(manifest_rows)} entries")


if __name__ == '__main__':
    main()
