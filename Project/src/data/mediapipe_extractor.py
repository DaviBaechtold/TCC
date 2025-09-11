import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import mediapipe as mp  # type: ignore
except Exception as e:
    mp = None

@dataclass
class ExtractConfig:
    max_frames: Optional[int] = None
    stride: int = 1
    hands: int = 1  # 1 or 2
    static_image_mode: bool = False

class MediaPipeHandExtractor:
    """Wrapper around MediaPipe Hands to produce per-frame 2D keypoints.

    Returns an array of shape [T, H, K, 2] where H=hands (1 or 2), K=21.
    If fewer hands are detected, missing entries are filled with NaN.
    """
    def __init__(self, cfg: ExtractConfig):
        if mp is None:
            raise ImportError("mediapipe not installed. Please pip install mediapipe")
        self.cfg = cfg
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=cfg.static_image_mode,
            max_num_hands=cfg.hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        frames = []
        count = 0
        k = 21
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % self.cfg.stride != 0:
                count += 1
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            # initialize as NaNs
            coords = np.full((self.cfg.hands, k, 2), np.nan, dtype=np.float32)
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for i, lm in enumerate(results.multi_hand_landmarks[: self.cfg.hands]):
                    for j, p in enumerate(lm.landmark[:k]):
                        coords[i, j, 0] = p.x * w
                        coords[i, j, 1] = p.y * h
            frames.append(coords)
            count += 1
            if self.cfg.max_frames and len(frames) >= self.cfg.max_frames:
                break
        cap.release()
        if len(frames) == 0:
            return np.empty((0, self.cfg.hands, 21, 2), dtype=np.float32)
        return np.stack(frames, axis=0)

    def extract_image_sequence(self, image_paths: List[str]) -> np.ndarray:
        """Extract landmarks from a list of image file paths.

        Returns an array of shape [T, H, 21, 2] where H is number of hands.
        Honors max_frames and stride from config.
        """
        frames = []
        k = 21
        count = 0
        # Ensure deterministic order
        for i, img_path in enumerate(sorted(image_paths)):
            if count % self.cfg.stride != 0:
                count += 1
                continue
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                # skip broken/missing files
                count += 1
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            h, w = image_bgr.shape[:2]
            coords = np.full((self.cfg.hands, k, 2), np.nan, dtype=np.float32)
            if results.multi_hand_landmarks:
                for hi, lm in enumerate(results.multi_hand_landmarks[: self.cfg.hands]):
                    for j, p in enumerate(lm.landmark[:k]):
                        coords[hi, j, 0] = p.x * w
                        coords[hi, j, 1] = p.y * h
            frames.append(coords)
            count += 1
            if self.cfg.max_frames and len(frames) >= self.cfg.max_frames:
                break
        if len(frames) == 0:
            return np.empty((0, self.cfg.hands, 21, 2), dtype=np.float32)
        return np.stack(frames, axis=0)


def save_npz(sequence: np.ndarray, out_path: str, label: Optional[str] = None):
    out = {"sequence": sequence}
    if label is not None:
        out["label"] = label
    np.savez_compressed(out_path, **out)


def load_npz(path: str) -> Tuple[np.ndarray, Optional[str]]:
    data = np.load(path, allow_pickle=True)
    seq = data["sequence"].astype(np.float32)
    label = data["label"].item() if "label" in data else None
    return seq, label
