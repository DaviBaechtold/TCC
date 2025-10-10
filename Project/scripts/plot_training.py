"""Small utility to plot training curves (loss/metrics) from mmengine/mmcv logs.

Supports: JSON logs produced by MMEngine (e.g. work_dirs/*/*.log.json) and TensorBoard event files.

Usage:
    python scripts/plot_training.py --logdir work_dirs/test_minimal5 --out plots/training_curves.png
"""
import argparse
import os
import json
import glob
import matplotlib.pyplot as plt


def find_json_logs(logdir):
    patterns = [os.path.join(logdir, "**", "*.log.json"), os.path.join(logdir, "*.log.json")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(list(set(files)))


def find_text_logs(logdir):
    patterns = [os.path.join(logdir, "**", "*.log"), os.path.join(logdir, "*.log")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # prefer mmengine timestamped folders
    return sorted(list(set(files)))


def parse_mmengine_json(logfile):
    iters = []
    losses = []
    metrics = {}
    with open(logfile, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # mmengine emits dicts with keys like "epoch", "iter", "loss", or nested metrics
            if "iter" in obj and ("loss" in obj or any(k.startswith("loss") for k in obj.keys())):
                iters.append(obj.get("iter", len(iters)))
                # prefer 'loss' key
                losses.append(obj.get("loss", obj.get("loss_rpn", None)))
            # collect other scalar metrics
            for k, v in obj.items():
                if isinstance(v, (int, float)) and k not in ("iter", "epoch", "time", "lr"):
                    metrics.setdefault(k, []).append((obj.get("iter", len(iters)-1), v))
    return iters, losses, metrics


def parse_mmengine_text(logfile):
    """Parse plain mmengine text log lines containing 'Epoch(train)' and 'loss:'
    Returns (iters, losses, metrics) where iters is pseudo-iter index (line idx).
    """
    iters = []
    losses = []
    metrics = {}
    with open(logfile, "r") as f:
        line_idx = 0
        for line in f:
            line_idx += 1
            if 'Epoch(train)' in line and 'loss:' in line:
                # example line fragment:
                # Epoch(train)  [1][  10/4682]  lr: 3.6092e-05  eta: ...  loss: 13.0077  loss_kpt: 13.0077  acc_pose: 0.0056
                try:
                    parts = line.split()
                    # find 'loss:' token and take next token as value
                    for i, tok in enumerate(parts):
                        if tok.startswith('loss:'):
                            # token may be 'loss:' or 'loss:13.0' (handle both)
                            if tok == 'loss:' and i+1 < len(parts):
                                val = float(parts[i+1])
                            else:
                                val = float(tok.split('loss:')[-1])
                            losses.append(val)
                            iters.append(line_idx)
                        # capture other scalar metrics like acc_pose
                        if tok.startswith('acc_') or tok.endswith('acc_pose:') or tok.endswith('acc_pose'):
                            # try to read next token
                            if tok.endswith(':') and i+1 < len(parts):
                                mval = float(parts[i+1])
                            else:
                                # token might be 'acc_pose:0.123' or similar
                                mval = float(tok.split(':')[-1])
                            metrics.setdefault('acc_pose', []).append((line_idx, mval))
                except Exception:
                    continue
    return iters, losses, metrics


def plot_curves(all_iters, all_losses, all_metrics, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    # Loss
    if any(len(l) for l in all_losses):
        for i, losses in enumerate(all_losses):
            plt.plot(all_iters[i], losses, label=f"loss_{i}")
    plt.xlabel("iter")
    plt.ylabel("loss / metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved combined plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Path to work_dir or logs folder")
    parser.add_argument("--out", required=True, help="Output image path")
    args = parser.parse_args()

    json_logs = find_json_logs(args.logdir)
    text_logs = find_text_logs(args.logdir)

    all_iters = []
    all_losses = []
    all_metrics = {}

    if json_logs:
        for j in json_logs:
            iters, losses, metrics = parse_mmengine_json(j)
            if iters:
                all_iters.append(iters)
                all_losses.append(losses)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append((j, v))

    # fallback to plain text logs if no json present or to add more series
    if text_logs:
        for t in text_logs:
            iters, losses, metrics = parse_mmengine_text(t)
            if iters:
                all_iters.append(iters)
                all_losses.append(losses)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append((t, v))

    if not all_iters and not all_metrics:
        print("No recognizable logs found under", args.logdir)
        return

    plot_curves(all_iters, all_losses, all_metrics, args.out)


if __name__ == "__main__":
    main()
