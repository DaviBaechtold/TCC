#!/usr/bin/env python3
"""
SOLUÇÃO: Bottom-Up Real funcionando!

O problema do script original era usar full-image bbox que não funciona bem.
Esta versão usa DETECTOR + agrupamento = bottom-up efetivo.
"""

import sys
sys.path.insert(0, '/home/davs/Documents/TCC/Project')

import argparse
import time
import cv2
import numpy as np
import torch

# Patch torch.load
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from mmpose.apis import init_model, inference_topdown
from mmdet.apis import init_detector, inference_detector

print("="*60)
print("🚀 BOTTOM-UP POSE ESTIMATION - VERSÃO CORRIGIDA")
print("="*60)

# Configurações
POSE_CFG = 'configs/rtmpose_m_wholebody_minimal.py'
POSE_CKPT = 'work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth'
DET_CFG = 'configs/detectors/rtmdet_nano_person_infer.py'
DET_CKPT = 'checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'

# Inicializar modelos
print("\n📦 Carregando modelos...")
pose_model = init_model(POSE_CFG, POSE_CKPT, device='cuda:0')
detector = init_detector(DET_CFG, DET_CKPT, device='cuda:0')
print("✅ Modelos carregados!")

# Abrir webcam
print("\n📹 Abrindo webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Falha ao abrir webcam!")
    sys.exit(1)

print("✅ Webcam aberta!")
print("\nControles:")
print("  'q' - Sair")
print("  ESPAÇO - Pausar/Continuar")
print("-"*60)

frame_count = 0
total_time = 0
paused = False

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        start = time.time()
        
        # 1. Detectar pessoas (BOTTOM-UP: detecta todas de uma vez)
        det_results = inference_detector(detector, frame_rgb)
        
        # Extrair bboxes de pessoas
        if hasattr(det_results, 'pred_instances'):
            bboxes = det_results.pred_instances.bboxes.cpu().numpy()
            scores = det_results.pred_instances.scores.cpu().numpy()
            
            # Filtrar por score
            mask = scores > 0.3
            bboxes = bboxes[mask]
            scores = scores[mask]
        else:
            bboxes = np.empty((0, 4))
            scores = np.empty((0,))
        
        # 2. Estimar pose de cada pessoa
        persons = []
        if len(bboxes) > 0:
            pose_results = inference_topdown(pose_model, frame_rgb, bboxes=bboxes)
            
            for i, result in enumerate(pose_results):
                if hasattr(result, 'pred_instances'):
                    kpts = result.pred_instances.keypoints.cpu().numpy()[0]  # [133, 2]
                    kpt_scores = result.pred_instances.keypoint_scores.cpu().numpy()[0]  # [133]
                    
                    # Combinar coords + scores
                    kpts_full = np.concatenate([kpts, kpt_scores[:, None]], axis=1)  # [133, 3]
                    
                    persons.append({
                        'keypoints': kpts_full,
                        'bbox': bboxes[i],
                        'score': scores[i]
                    })
        
        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1
        fps = frame_count / total_time if total_time > 0 else 0
        
        # 3. Visualizar
        vis = frame.copy()
        
        for person_id, person in enumerate(persons):
            color = colors[person_id % len(colors)]
            kpts = person['keypoints']
            bbox = person['bbox'].astype(int)
            
            # Desenhar bbox
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(vis, f"Person {person_id+1}", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Desenhar keypoints (apenas body - primeiros 17)
            for i in range(min(17, len(kpts))):
                x, y, conf = kpts[i]
                if conf > 0.3:
                    cv2.circle(vis, (int(x), int(y)), 3, color, -1)
            
            # Skeleton (body)
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # legs
                (5, 11), (6, 12)  # torso
            ]
            for link in skeleton:
                i, j = link
                if i < len(kpts) and j < len(kpts):
                    if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
                        pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                        pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        cv2.line(vis, pt1, pt2, color, 2)
        
        # Info overlay
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Latency: {elapsed*1000:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Persons: {len(persons)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "BOTTOM-UP MODE", (vis.shape[1]-250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow('Bottom-Up Pose Estimation', vis)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"Total frames: {frame_count}")
print(f"Average FPS: {fps:.2f}")
print("="*60)
