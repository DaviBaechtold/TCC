# Temporal Transformer-Based Gesture Recognition (2D Skeletons)

This project recognizes dynamic hand/body gestures from monocular RGB videos. It extracts 2D keypoints with MediaPipe, builds temporal sequences, and classifies them using a Transformer encoder. Includes a baseline (LSTM/MLP), training loop, evaluation, and a simple real-time demo.

## Layout
- `src/data/`: data extraction and datasets
- `src/models/`: transformer, baselines
- `src/utils/`: training, metrics, viz
- `scripts/`: CLI entry-points

## Quickstart
1. Create a Python env and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Project/requirements.txt
```
2. Extract skeletons from a video folder (mp4/avi) to `.npz` sequences:
```bash
python Project/scripts/extract_keypoints.py --videos /path/to/videos --out Project/data/npz --max-frames 120
```
3. Train the Transformer on the generated dataset manifest:
```bash
python Project/scripts/train.py --config Project/configs/transformer.yaml
```
4. Quick camera test (show face/hand/pose landmarks and bounding boxes):
```bash
python Project/scripts/camera_test.py --mirror
```

5. Run the real-time classification demo (requires a trained checkpoint):
```bash
python Project/scripts/demo.py --checkpoint runs/transformer/best.pt
```

See `Project/configs/` for hyperparameters.
