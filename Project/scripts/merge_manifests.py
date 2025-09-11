#!/usr/bin/env python
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='Input manifest CSVs (each with header path,label)')
    ap.add_argument('--out', required=True, help='Output merged manifest CSV')
    ap.add_argument('--dedup', action='store_true', help='Deduplicate by path')
    args = ap.parse_args()

    seen = set()
    rows = []
    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"WARN: missing manifest {p}, skipping")
            continue
        with p.open('r', newline='') as f:
            r = csv.reader(f)
            header = next(r, None)
            # accept any header, expect columns path,label
            for row in r:
                if len(row) < 2:
                    continue
                key = row[0]
                if args.dedup:
                    if key in seen:
                        continue
                    seen.add(key)
                rows.append((row[0], row[1]))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'])
        w.writerows(rows)
    print(f"Wrote merged manifest {outp} with {len(rows)} entries from {len(args.inputs)} inputs")


if __name__ == '__main__':
    main()
