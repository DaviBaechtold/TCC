#!/usr/bin/env python
"""
Demo de visualização 3D em tempo real usando apenas OpenCV.
Alternativa mais simples que não depende de matplotlib.

Uso:
    python Project/scripts/demo_3d_opencv.py \
        --checkpoint Project/data/lifter_runs/lifter_best.pt \
        --camera 0 --device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    import mediapipe as mp
except ImportError:
    mp = None

from src.models.lifter import build_lifter, root_center

# Conexões dos ossos para visualização (17 keypoints principais)
POSE_17_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]

# Conexões baseadas nos nossos índices 0-16
SKELETON_CONNECTIONS = [
    # Torso
    (7, 8),   # Left shoulder (7) to right shoulder (8) 
    (7, 13),  # Left shoulder (7) to left hip (13)
    (8, 14),  # Right shoulder (8) to right hip (14)
    (13, 14), # Left hip (13) to right hip (14)
    
    # Left arm
    (7, 9),   # Left shoulder (7) to left elbow (9)
    (9, 11),  # Left elbow (9) to left wrist (11)
    
    # Right arm  
    (8, 10),  # Right shoulder (8) to right elbow (10)
    (10, 12), # Right elbow (10) to right wrist (12)
    
    # Left leg
    (13, 15), # Left hip (13) to left knee (15)
    
    # Right leg
    (14, 16), # Right hip (14) to right knee/ankle (16)
    
    # Head connections
    (0, 1),   # Nose to left eye inner
    (0, 2),   # Nose to right eye inner  
    (1, 3),   # Left eye inner to left eye
    (2, 4),   # Right eye inner to right eye
    (3, 5),   # Left eye to left ear
    (4, 6),   # Right eye to right ear
]

class TemporalSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.poses_buffer: List[np.ndarray] = []
    
    def smooth(self, pose_3d: np.ndarray) -> np.ndarray:
        self.poses_buffer.append(pose_3d.copy())
        if len(self.poses_buffer) > self.window_size:
            self.poses_buffer.pop(0)
        
        if len(self.poses_buffer) == 1:
            return pose_3d
        
        weights = np.linspace(0.5, 1.0, len(self.poses_buffer))
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(pose_3d)
        for i, pose in enumerate(self.poses_buffer):
            smoothed += weights[i] * pose
            
        return smoothed

class Real3DDemo:
    def __init__(self, checkpoint_path: str, camera_id: int = 0, device: str = 'cpu', 
                 mirror: bool = True, smooth: bool = True):
        self.device = torch.device(device)
        self.mirror = mirror
        
        # Load lifter model
        print(f"Carregando modelo de {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = ckpt.get('cfg', {})
        self.num_joints = int(cfg.get('num_joints', 17))
        model_type = cfg.get('model', {}).get('type', 'mlp')
        
        print(f"Modelo: {model_type}, Joints: {self.num_joints}")
        
        self.model = build_lifter(model_type, self.num_joints)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {camera_id}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup MediaPipe
        if mp is None:
            raise ImportError('mediapipe não está instalado')
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Setup temporal smoothing
        self.smoother = TemporalSmoother(window_size=5) if smooth else None
        
        # State
        self.current_pose_2d = None
        self.current_pose_3d = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def extract_pose_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 2D keypoints using MediaPipe"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            pose_2d = []
            
            for idx in POSE_17_INDICES:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    x = lm.x * w
                    y = lm.y * h
                    pose_2d.append([x, y])
                else:
                    pose_2d.append([0.0, 0.0])
            
            return np.array(pose_2d, dtype=np.float32)
        else:
            return None
    
    def normalize_pose_2d(self, pose_2d: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Normalize 2D pose for the lifter model"""
        h, w = frame_shape[:2]
        
        pose_2d_norm = pose_2d.copy()
        pose_2d_norm[:, 0] -= w / 2
        pose_2d_norm[:, 1] -= h / 2
        
        scale = min(w, h) / 2
        pose_2d_norm /= scale
        
        return pose_2d_norm
    
    def lift_to_3d(self, pose_2d: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert 2D pose to 3D using trained lifter"""
        pose_2d_norm = self.normalize_pose_2d(pose_2d, frame_shape)
        
        x = torch.from_numpy(pose_2d_norm[None]).to(self.device)
        x = root_center(x, 0)
        
        with torch.no_grad():
            pose_3d = self.model(x).cpu().numpy()[0]
            
        if self.smoother:
            pose_3d = self.smoother.smooth(pose_3d)
            
        return pose_3d
    
    def draw_skeleton_2d(self, pose_2d: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Draw 2D skeleton on frame"""
        frame_viz = frame.copy()
        
        # Draw joints
        for i, (x, y) in enumerate(pose_2d):
            if x > 0 and y > 0:
                cv2.circle(frame_viz, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if (start_idx < len(pose_2d) and end_idx < len(pose_2d)):
                pt1 = pose_2d[start_idx]
                pt2 = pose_2d[end_idx]
                
                if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                    cv2.line(frame_viz, 
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (255, 0, 0), 2)
                
        return frame_viz
    
    def draw_3d_projection(self, pose_3d: np.ndarray, canvas_size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """Draw 3D pose as 2D projection on canvas"""
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        # Simple orthographic projection (ignore Z for now, use X-Y)
        scale = 150
        center_x, center_y = canvas_size[0] // 2, canvas_size[1] // 2
        
        # Project 3D points to 2D
        points_2d = []
        for joint in pose_3d:
            x = int(center_x + joint[0] * scale)
            y = int(center_y - joint[1] * scale)  # Flip Y
            points_2d.append([x, y])
        
        points_2d = np.array(points_2d)
        
        # Draw joints
        for i, (x, y) in enumerate(points_2d):
            if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)
        
        # Draw connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(points_2d) and end_idx < len(points_2d):
                pt1 = tuple(points_2d[start_idx].astype(int))
                pt2 = tuple(points_2d[end_idx].astype(int))
                
                # Check bounds
                if (0 <= pt1[0] < canvas_size[0] and 0 <= pt1[1] < canvas_size[1] and
                    0 <= pt2[0] < canvas_size[0] and 0 <= pt2[1] < canvas_size[1]):
                    cv2.line(canvas, pt1, pt2, (0, 255, 255), 2)
        
        # Add title
        cv2.putText(canvas, "3D Pose Projection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return canvas
    
    def update_fps(self) -> float:
        """Update and return FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            return fps
        return 0
    
    def run(self):
        """Main demo loop"""
        print("Demo iniciado!")
        print("Controles:")
        print("  's' - Salvar pose 3D atual")
        print("  'q' ou ESC - Sair")
        
        cv2.namedWindow("Demo 3D Pose", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erro ao capturar frame da câmera")
                    break
                    
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                
                # Extract 2D pose
                pose_2d = self.extract_pose_keypoints(frame)
                
                if pose_2d is not None:
                    self.current_pose_2d = pose_2d
                    pose_3d = self.lift_to_3d(pose_2d, frame.shape)
                    self.current_pose_3d = pose_3d
                    
                    # Visualize 2D
                    frame_viz = self.draw_skeleton_2d(pose_2d, frame)
                    
                    # Create 3D projection
                    canvas_3d = self.draw_3d_projection(pose_3d)
                    
                    # Combine views side by side
                    # Resize frame to match height of 3D canvas
                    h_target = canvas_3d.shape[0]
                    w_target = int(frame_viz.shape[1] * h_target / frame_viz.shape[0])
                    frame_resized = cv2.resize(frame_viz, (w_target, h_target))
                    
                    # Concatenate horizontally
                    combined = np.hstack([frame_resized, canvas_3d])
                    
                else:
                    # No pose detected
                    frame_viz = frame.copy()
                    cv2.putText(frame_viz, "Pose nao detectada", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    combined = frame_viz
                
                # Show FPS
                fps = self.update_fps()
                if fps > 0:
                    cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Demo 3D Pose", combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    break
                elif key == ord('s') and self.current_pose_3d is not None:
                    filename = f'pose_3d_{int(time.time())}.npy'
                    np.save(filename, self.current_pose_3d)
                    print(f"Pose 3D salva em: {filename}")
                    
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Encerrando demo...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'pose'):
            self.pose.close()

def main():
    parser = argparse.ArgumentParser(description="Demo de visualização 3D em tempo real (OpenCV)")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Caminho para o checkpoint do lifter treinado')
    parser.add_argument('--camera', type=int, default=0,
                       help='Índice da câmera (padrão: 0)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device para inferência (cpu/cuda)')
    parser.add_argument('--no-mirror', action='store_true',
                       help='Desabilitar espelhamento da imagem')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Desabilitar suavização temporal')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Erro: Checkpoint não encontrado em {args.checkpoint}")
        print("Execute primeiro o treinamento:")
        print("python Project/scripts/train_lifter.py --config Project/configs/lifter.yaml --synthetic")
        return 1
    
    try:
        demo = Real3DDemo(
            checkpoint_path=args.checkpoint,
            camera_id=args.camera,
            device=args.device,
            mirror=not args.no_mirror,
            smooth=not args.no_smooth
        )
        demo.run()
    except Exception as e:
        print(f"Erro ao executar demo: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())