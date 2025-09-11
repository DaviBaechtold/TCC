#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
import os
import csv

# Ensure 'Project/src' is on sys.path when executing as a script
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mediapipe_extractor import MediaPipeHandExtractor, ExtractConfig, save_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True, help='Folder with videos')
    ap.add_argument('--out', required=True, help='Output folder for npz files')
    ap.add_argument('--labels-from-folders', action='store_true', help='Infer label as parent folder name')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--hands', type=int, default=1)
    ap.add_argument('--manifest', default='Project/data/manifest.csv')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExtractConfig(max_frames=args.max_frames, stride=args.stride, hands=args.hands)
    extractor = MediaPipeHandExtractor(cfg)

    video_paths = []
    root = Path(args.videos)
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
        video_paths.extend(root.rglob(ext))

    entries = []
    for v in sorted(video_paths):
        label = v.parent.name if args.labels_from_folders else 'unknown'
        seq = extractor.extract_video(str(v))
        out_path = out_dir / (v.stem + '.npz')
        save_npz(seq, str(out_path), label=label)
        entries.append((str(out_path), label))
        print(f"Saved {out_path} (T={seq.shape[0]})")

    # write manifest
    Path(Path(args.manifest).parent).mkdir(parents=True, exist_ok=True)
    with open(args.manifest, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        writer.writerows(entries)
    print(f"Wrote manifest with {len(entries)} entries to {args.manifest}")


if __name__ == '__main__':
    main()
