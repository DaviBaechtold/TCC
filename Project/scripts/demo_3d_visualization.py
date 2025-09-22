#!/usr/bin/env python
"""
Demo de visualização 3D em tempo real.
Captura keypoints 2D da webcam → Lifting para 3D → Visualização do esqueleto 3D.

Uso:
    python Project/scripts/demo_3d_visualization.py \
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

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for GUI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    import mediapipe as mp
except ImportError:
    mp = None

from src.models.lifter import build_lifter, root_center

# Conexões dos ossos para visualização (17 keypoints principais)
# Mapeamento para índices do nosso subset de 17 pontos:
# MediaPipe original -> Nosso índice 17-joint
POSE_17_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
#                  [0, 1, 2, 3, 4, 5, 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16] (nossos índices)

# Conexões baseadas nos nossos índices 0-16
SKELETON_CONNECTIONS = [
    # Torso (usando nossos índices remapeados)
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
    (15, 16), # Left knee (15) to left ankle (16) - Note: só temos até 16
    
    # Right leg
    (14, 16), # Right hip (14) to right knee/ankle (16) - simplificado
    
    # Head connections (face keypoints)
    (0, 1),   # Nose to left eye inner
    (0, 2),   # Nose to right eye inner  
    (1, 3),   # Left eye inner to left eye
    (2, 4),   # Right eye inner to right eye
    (3, 5),   # Left eye to left ear
    (4, 6),   # Right eye to right ear
]

class TemporalSmoother:
    """Suavização temporal para poses 3D"""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.poses_buffer: List[np.ndarray] = []
    
    def smooth(self, pose_3d: np.ndarray) -> np.ndarray:
        """Aplica suavização temporal usando média móvel"""
        self.poses_buffer.append(pose_3d.copy())
        if len(self.poses_buffer) > self.window_size:
            self.poses_buffer.pop(0)
        
        if len(self.poses_buffer) == 1:
            return pose_3d
        
        # Média ponderada dando mais peso às poses mais recentes
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
            raise ImportError('mediapipe não está instalado. Execute: pip install mediapipe')
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Setup temporal smoothing
        self.smoother = TemporalSmoother(window_size=5) if smooth else None
        
        # Setup 3D plot
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(15, 7))
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        
        self.setup_3d_plot()
        
        # Show the figure initially
        plt.show(block=False)
        plt.draw()
        
        # State
        self.current_pose_2d = None
        self.current_pose_3d = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def setup_3d_plot(self):
        """Configure 3D plot appearance"""
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y') 
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-1, 1])
        self.ax_3d.set_zlim([-1, 1])
        self.ax_3d.set_title('3D Pose (Lifted)')
        
    def extract_pose_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 2D keypoints using MediaPipe"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        
        if result.pose_landmarks:
            # Extract landmarks for the 17 keypoints we need
            landmarks = result.pose_landmarks.landmark
            pose_2d = []
            
            for idx in POSE_17_INDICES:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = lm.x * w
                    y = lm.y * h
                    pose_2d.append([x, y])
                else:
                    # Fill with zeros if landmark is missing
                    pose_2d.append([0.0, 0.0])
            
            return np.array(pose_2d, dtype=np.float32)
        else:
            # Return None if no pose detected
            return None
    
    def normalize_pose_2d(self, pose_2d: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Normalize 2D pose for the lifter model"""
        h, w = frame_shape[:2]
        
        # Center and normalize
        pose_2d_norm = pose_2d.copy()
        pose_2d_norm[:, 0] -= w / 2    # center x
        pose_2d_norm[:, 1] -= h / 2    # center y
        
        # Normalize by image scale (use smaller dimension)
        scale = min(w, h) / 2
        pose_2d_norm /= scale
        
        return pose_2d_norm
    
    def lift_to_3d(self, pose_2d: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert 2D pose to 3D using trained lifter"""
        # Normalize pose
        pose_2d_norm = self.normalize_pose_2d(pose_2d, frame_shape)
        
        # Convert to tensor and apply root centering
        x = torch.from_numpy(pose_2d_norm[None]).to(self.device)  # Add batch dim
        x = root_center(x, 0)  # Root center using joint 0
        
        with torch.no_grad():
            pose_3d = self.model(x).cpu().numpy()[0]  # Remove batch dim
            
        # Apply temporal smoothing if enabled
        if self.smoother:
            pose_3d = self.smoother.smooth(pose_3d)
            
        return pose_3d
    
    def draw_skeleton_2d(self, pose_2d: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Draw 2D skeleton on frame"""
        frame_viz = frame.copy()
        
        # Draw joints
        for i, (x, y) in enumerate(pose_2d):
            if x > 0 and y > 0:  # Only draw valid points
                cv2.circle(frame_viz, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame_viz, str(i), (int(x)+5, int(y)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            # Use direct indices since SKELETON_CONNECTIONS is already mapped to our 17-joint system
            if (start_idx < len(pose_2d) and end_idx < len(pose_2d)):
                pt1 = pose_2d[start_idx]
                pt2 = pose_2d[end_idx]
                
                # Only draw if both points are valid
                if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                    cv2.line(frame_viz, 
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (255, 0, 0), 2)
                
        return frame_viz
    
    def draw_skeleton_3d(self, pose_3d: np.ndarray):
        """Draw 3D skeleton"""
        self.ax_3d.clear()
        self.setup_3d_plot()
        
        # Plot joints
        xs, ys, zs = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
        self.ax_3d.scatter(xs, ys, zs, c='red', s=60, alpha=0.8)
        
        # Label some key joints
        key_joints = [0, 7, 8, 13, 14, 15, 16]  # nose, shoulders, hips, etc.
        for i in key_joints:
            if i < len(pose_3d):
                self.ax_3d.text(xs[i], ys[i], zs[i], str(i), fontsize=8)
        
        # Plot connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                self.ax_3d.plot([xs[start_idx], xs[end_idx]], 
                               [ys[start_idx], ys[end_idx]], 
                               [zs[start_idx], zs[end_idx]], 
                               'b-', linewidth=2, alpha=0.7)
        
        # Set viewing angle for better visualization
        self.ax_3d.view_init(elev=20, azim=45)
        
    def update_fps(self) -> float:
        """Update and return FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            return fps
        return 0
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'r':  # Reset 3D view
            self.ax_3d.view_init(elev=20, azim=45)
        elif event.key == 's' and self.current_pose_3d is not None:  # Save current pose
            filename = f'pose_3d_{int(time.time())}.npy'
            np.save(filename, self.current_pose_3d)
            print(f"Pose 3D salva em: {filename}")
        elif event.key == 'q':  # Quit
            self.cleanup()
            return False
        return True
    
    def run(self):
        """Main demo loop"""
        print("Demo iniciado!")
        print("Controles:")
        print("  'r' - Reset visualização 3D")
        print("  's' - Salvar pose 3D atual")
        print("  'q' - Sair")
        print("  ESC - Sair")
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erro ao capturar frame da câmera")
                    break
                    
                # Mirror for selfie view
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                
                # Extract 2D pose
                pose_2d = self.extract_pose_keypoints(frame)
                
                if pose_2d is not None:
                    # Store for later use
                    self.current_pose_2d = pose_2d
                    
                    # Lift to 3D
                    pose_3d = self.lift_to_3d(pose_2d, frame.shape)
                    self.current_pose_3d = pose_3d
                    
                    # Visualize 2D
                    frame_viz = self.draw_skeleton_2d(pose_2d, frame)
                    
                    # Update 3D view
                    self.draw_skeleton_3d(pose_3d)
                    
                else:
                    # No pose detected
                    frame_viz = frame.copy()
                    cv2.putText(frame_viz, "Pose nao detectada", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update 2D view
                self.ax_2d.clear()
                self.ax_2d.imshow(cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB))
                self.ax_2d.set_title('2D Pose Detection')
                self.ax_2d.axis('off')
                
                # Show FPS
                fps = self.update_fps()
                if fps > 0:
                    self.fig.suptitle(f'Demo 3D Pose - FPS: {fps:.1f}')
                
                # Force update and show the figure
                plt.draw()
                plt.show(block=False)
                plt.pause(0.001)
                
                # Check for quit key in OpenCV (backup)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    break
                    
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
        plt.close('all')
        if hasattr(self, 'pose'):
            self.pose.close()

def main():
    parser = argparse.ArgumentParser(description="Demo de visualização 3D em tempo real")
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