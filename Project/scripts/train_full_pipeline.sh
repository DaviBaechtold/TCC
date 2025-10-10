#!/bin/bash

# Simple wrapper to start RTMPose training on the grayscale dataset

set -e

CONFIG=${CONFIG:-"configs/rtmpose_s_grayscale_rt.py"}
WORK_DIR=${WORK_DIR:-"work_dirs/baseline_grayscale"}
AMP=${AMP:-1}
LOAD_FROM=${LOAD_FROM:-""}

echo "=================================="
echo "Train Full Pipeline"
echo "=================================="
echo "Config:     $CONFIG"
echo "Work dir:   $WORK_DIR"
if [ -n "$LOAD_FROM" ]; then
  echo "Load from:  $LOAD_FROM"
fi
echo "AMP:        $AMP"
echo "=================================="

mkdir -p "$WORK_DIR"

CMD=(python src/training/train_pose.py \
  --config "$CONFIG" \
  --work-dir "$WORK_DIR")

if [ "$AMP" = "1" ] || [ "$AMP" = "true" ]; then
  CMD+=(--amp)
fi

if [ -n "$LOAD_FROM" ]; then
  CMD+=(--load-from "$LOAD_FROM")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "\n✓ Training launched. Check logs in: $WORK_DIR"
