"""Run RTMPose inference on RGB and IR images and compare results."""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

# Register safe globals for torch.load
for typ in [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.str_]:
    torch.serialization.add_safe_globals([typ])

# Monkeypatch torch.load
_original_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(f, *args, **kwargs)
torch.load = _patched_load

try:
    from mmpose.apis import init_model, inference_topdown
except ImportError as e:
    print(f"Error: {e}")
    print("Please install mmpose")
    sys.exit(1)

try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    init_detector = None
    inference_detector = None

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None

# Try to import WholeBody OKS sigmas from MMPose for correct AP/AR computation
WHOLEBODY_OKS_SIGMAS = None
# Default COCO-17 OKS sigmas
DEFAULT_COCO17_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087,
    0.089, 0.089
], dtype=np.float32)
try:
    # mmpose>=1.2
    from mmpose.evaluation.metrics.coco_wholebody_metric import (
        COCO_WHOLEBODY_KEYPOINTS_SIGMAS as WHOLEBODY_OKS_SIGMAS,
    )
except Exception:
    try:
        # fallback older path (in case of different package layout)
        from mmpose.evaluation.functional.keypoint_evaluation import (
            COCO_WHOLEBODY_KEYPOINTS_SIGMAS as WHOLEBODY_OKS_SIGMAS,
        )
    except Exception:
        WHOLEBODY_OKS_SIGMAS = None


def extract_kps(result):
    """Extract keypoints from PoseDataSample."""
    if not result or len(result) == 0:
        return None
    collected = []

    for sample in result:
        if hasattr(sample, 'pred_instances'):
            instances = sample.pred_instances
            if hasattr(instances, 'keypoints'):
                keypoints = instances.keypoints
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                else:
                    keypoints = np.asarray(keypoints)
                if keypoints.ndim == 2:
                    keypoints = keypoints[None, ...]
                collected.append(keypoints)

    if not collected:
        return None

    return np.concatenate(collected, axis=0)


def _collect_coco_results(result, image_id, expected_kpt_num=None):
    """Convert PoseDataSample list into COCO-format prediction dicts.

    Args:
        result: output from inference_topdown (list[PoseDataSample])
        image_id: int COCO image id

    Returns:
        list of dicts ready for COCO.loadRes
    """
    preds = []
    if not result:
        return preds

    for sample in result:
        if not hasattr(sample, 'pred_instances'):
            continue
        inst = sample.pred_instances

        # keypoints: (N, K, 2 or 3)
        if not hasattr(inst, 'keypoints'):
            continue
        kps = inst.keypoints
        try:
            kps = kps.cpu().numpy()
        except Exception:
            kps = np.asarray(kps)

        # optional per-keypoint scores
        kp_scores = None
        if hasattr(inst, 'keypoint_scores'):
            try:
                kp_scores = inst.keypoint_scores.cpu().numpy()
            except Exception:
                kp_scores = np.asarray(inst.keypoint_scores)

        # optional instance detection scores
        det_scores = None
        if hasattr(inst, 'scores'):
            try:
                det_scores = inst.scores.cpu().numpy()
            except Exception:
                det_scores = np.asarray(inst.scores)

        # Normalize shapes
        if kps.ndim == 2:
            kps = kps[None, ...]

        # Adjust to expected number of keypoints (e.g., map 133→17 body-only)
        if expected_kpt_num is not None:
            if kps.shape[1] > expected_kpt_num:
                # Assume first 17 in WholeBody correspond to COCO body17
                kps = kps[:, :expected_kpt_num, :]
                if kp_scores is not None:
                    kp_scores = kp_scores[:, :expected_kpt_num]
            elif kps.shape[1] < expected_kpt_num:
                # Pad missing keypoints with zeros (rare)
                pad_n = expected_kpt_num - kps.shape[1]
                pad_xy = np.zeros((kps.shape[0], pad_n, kps.shape[2]), dtype=kps.dtype)
                kps = np.concatenate([kps, pad_xy], axis=1)
                if kp_scores is not None:
                    pad_s = np.zeros((kp_scores.shape[0], pad_n), dtype=kp_scores.dtype)
                    kp_scores = np.concatenate([kp_scores, pad_s], axis=1)

        N, K = kps.shape[0], kps.shape[1]
        for n in range(N):
            xy = kps[n]
            if xy.shape[-1] == 2:
                # if only (x, y), set visibility to 2
                vis = np.full((K, 1), 2.0, dtype=np.float32)
                kpt = np.concatenate([xy, vis], axis=-1)
            else:
                # (x, y, score/vis). Convert to COCO v (0,1,2). Use 2 if score>0.
                v = (xy[:, 2] > 0).astype(np.float32) * 2.0
                kpt = np.stack([xy[:, 0], xy[:, 1], v], axis=-1)

            # score for the detection (fallback to mean kp score)
            if det_scores is not None and n < len(det_scores):
                score = float(det_scores[n])
            elif kp_scores is not None and kp_scores.ndim >= 2:
                score = float(np.mean(kp_scores[n]))
            elif xy.shape[-1] == 3:
                score = float(np.mean(xy[:, 2]))
            else:
                score = 1.0

            preds.append({
                'image_id': int(image_id),
                'category_id': 1,  # person
                'keypoints': kpt.reshape(-1).tolist(),
                'score': score,
            })

    return preds


def normalized_keypoint_distance(kp1, kp2):
    """Compute normalized distance between two keypoint sets."""
    if kp1 is None or kp2 is None:
        return float('inf')
    
    if kp1.ndim == 3:
        kp1 = kp1[0]
    if kp2.ndim == 3:
        kp2 = kp2[0]
    
    # Normalize by torso distance (shoulders: keypoints 5-6)
    torso_dist = np.linalg.norm(kp1[5, :2] - kp1[6, :2])
    if torso_dist < 1e-6:
        torso_dist = 1.0
    
    dists = np.linalg.norm(kp1[:, :2] - kp2[:, :2], axis=1)
    mean_dist = np.mean(dists)
    
    return mean_dist / torso_dist


def main():
    parser = argparse.ArgumentParser(description='Evaluate RTMPose on RGB vs IR')
    parser.add_argument('--cfg', type=str, 
                        default='work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py')
    parser.add_argument('--ckpt', type=str,
                        default='work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth')
    parser.add_argument('--rgb-dir', type=str, default='data/raw/val2017')
    parser.add_argument('--ir-dir', type=str, default='data/processed/grayscale/val2017')
    parser.add_argument('--ann-file', type=str,
                        default='data/processed/grayscale/annotations/coco_wholebody_val_v1.0.json',
                        help='COCO-format annotation file with bboxes')
    parser.add_argument('--det-cfg', type=str, default=None,
                        help='Detector config (e.g. RTMDet) for bbox generation')
    parser.add_argument('--det-ckpt', type=str, default=None,
                        help='Detector checkpoint path')
    parser.add_argument('--det-score-thr', type=float, default=0.4,
                        help='Score threshold for detector boxes')
    parser.add_argument('--out-dir', type=str, default='work_dirs/eval_results')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--force-k', type=int, choices=[17, 133], default=None,
                        help='Força o número de keypoints esperado (17 ou 133) apenas para mapeamento e sigmas. '
                             'Se não bater com o ann-file, será ignorado com aviso.')
    parser.add_argument('--wholebody-eval', action='store_true',
                    help='Se o ann-file for COCO-WholeBody (com foot/face/hand kpts), '
                        'constrói um GT temporário com 133 keypoints (body+foot+face+hands) '
                        'para calcular Whole AP/AR com COCOeval.')
    
    args = parser.parse_args()

    if args.det_cfg and args.det_ckpt:
        if init_detector is None or inference_detector is None:
            print('❌ mmdet is required for detector-based bboxes, but it is not installed.')
            sys.exit(1)
        det_model = init_detector(args.det_cfg, args.det_ckpt, device=args.device)
        use_detector = True
    else:
        det_model = None
        use_detector = False

    if not use_detector:
        if args.ann_file and not Path(args.ann_file).is_file():
            print(f"❌ Annotation file not found: {args.ann_file}")
            sys.exit(1)

        if args.ann_file and COCO is None:
            print('❌ pycocotools not installed; please install it or omit --ann-file')
            sys.exit(1)
    
    # Create output directories
    os.makedirs(os.path.join(args.out_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'ir'), exist_ok=True)
    
    print(f"Loading model from {args.ckpt}...")
    
    # Initialize model
    model = init_model(args.cfg, args.ckpt, device=args.device)
    print("Model loaded successfully!")
    
    # Build bbox lookup from annotations if available
    bbox_lookup = {}
    gt_kpt_num = None
    if not use_detector and args.ann_file:
        ann_path = args.ann_file

        # Carrega JSON cru para opcionalmente montar GT 133 kpts
        coco = None
        if args.wholebody_eval:
            try:
                with open(ann_path, 'r') as f:
                    raw = json.load(f)
                anns = raw.get('annotations', [])
                # Detecta campos wholebody auxiliares
                has_wb = any(('foot_kpts' in a or 'face_kpts' in a or 'lefthand_kpts' in a or 'righthand_kpts' in a) for a in anns)
                if has_wb:
                    # Constrói novo dataset com keypoints=133*3
                    new_ds = dict(raw)  # shallow copy ok, vamos sobrescrever chaves
                    new_anns = []
                    for a in anns:
                        body = a.get('keypoints', [0]*51)
                        foot = a.get('foot_kpts', [0]*18)
                        face = a.get('face_kpts', [0]*204)
                        lhand = a.get('lefthand_kpts', [0]*63)
                        rhand = a.get('righthand_kpts', [0]*63)
                        combined = body + foot + face + lhand + rhand
                        a2 = dict(a)
                        a2['keypoints'] = combined
                        # num_keypoints = número de visíveis (>0)
                        vis = 0
                        for i in range(2, len(combined), 3):
                            if combined[i] > 0:
                                vis += 1
                        a2['num_keypoints'] = vis
                        # Remove campos auxiliares para evitar ambiguidade
                        for k in ('foot_kpts','face_kpts','lefthand_kpts','righthand_kpts'):
                            if k in a2:
                                del a2[k]
                        new_anns.append(a2)

                    new_ds['annotations'] = new_anns
                    # Atualiza categories->keypoints para 133 nomes genéricos, se necessário
                    cats = new_ds.get('categories', [])
                    for c in cats:
                        kp = c.get('keypoints', [])
                        if not isinstance(kp, list) or len(kp) != 133:
                            c['keypoints'] = [f'kpt_{i}' for i in range(133)]
                            c['skeleton'] = c.get('skeleton', [])

                    # Salva temporário e carrega com COCO
                    os.makedirs(args.out_dir, exist_ok=True)
                    tmp_ann = os.path.join(args.out_dir, 'ann_wholebody_133_tmp.json')
                    with open(tmp_ann, 'w') as f:
                        json.dump(new_ds, f)
                    coco = COCO(tmp_ann)
                    gt_kpt_num = 133
                else:
                    coco = COCO(ann_path)
            except Exception as e:
                print(f"[warn] wholebody-eval falhou ({e}); usando ann-file original.")
                coco = COCO(ann_path)
        else:
            coco = COCO(ann_path)
        # Determine GT keypoint count (17 vs 133)
        try:
            cat_ids = coco.getCatIds(catNms=['person'])
            cats = coco.loadCats(cat_ids)
            for c in cats:
                if 'keypoints' in c and isinstance(c['keypoints'], list) and len(c['keypoints']) > 0:
                    gt_kpt_num = int(len(c['keypoints']))
                    break
        except Exception:
            gt_kpt_num = None
        if gt_kpt_num is None:
            # Fallback: inspect first annotation
            ann_ids_all = coco.getAnnIds()
            if ann_ids_all:
                ann0 = coco.loadAnns([ann_ids_all[0]])[0]
                if 'keypoints' in ann0:
                    gt_kpt_num = int(len(ann0['keypoints']) // 3)

        img_ids = coco.getImgIds()
        name_to_id = {}
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            name_to_id[info['file_name']] = img_id
        for fname, img_id in name_to_id.items():
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
            if boxes:
                bbox_lookup[fname] = np.array(boxes, dtype=np.float32)
    else:
        coco = None

    # Get image list
    rgb_images = sorted([f for f in os.listdir(args.rgb_dir) if f.endswith(('.jpg', '.png'))])
    if not use_detector and args.ann_file:
        rgb_images = [f for f in rgb_images if f in bbox_lookup]
    rgb_images = rgb_images[:args.n]
    
    print(f"Processing {len(rgb_images)} images...")
    
    distances = []
    per_person_distances = []
    
    # Effective K: respect ann-file; honor --force-k only if it matches
    effective_k = gt_kpt_num
    if args.force_k is not None:
        if gt_kpt_num is not None and args.force_k != gt_kpt_num:
            print(f"[warn] --force-k={args.force_k} ignora do pois ann-file tem K={gt_kpt_num}; usando K={gt_kpt_num}.")
        else:
            effective_k = args.force_k

    # Collect predictions (for concise AP/AR metrics)
    coco_preds_ir = []
    coco_preds_rgb = []

    for i, img_name in enumerate(rgb_images):
        print(f"[{i+1}/{len(rgb_images)}] Processing {img_name}...")
        
        rgb_path = os.path.join(args.rgb_dir, img_name)
        ir_path = os.path.join(args.ir_dir, img_name)
        
        if not os.path.exists(ir_path):
            print(f"  Warning: IR image not found: {ir_path}")
            continue
        
        # Read images
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        if rgb_img is None or ir_img is None:
            print(f"  Warning: Could not read images")
            continue
        
        if use_detector:
            det_result = inference_detector(det_model, rgb_img)
            boxes = []
            if hasattr(det_result, 'pred_instances'):
                inst = det_result.pred_instances
                b = inst.bboxes.cpu().numpy()
                s = inst.scores.cpu().numpy()
                keep = s >= args.det_score_thr
                boxes = b[keep]
            else:
                det_array = det_result[0]  # person class
                if det_array.size > 0:
                    keep = det_array[:, 4] >= args.det_score_thr
                    boxes = det_array[keep, :4]
            if len(boxes) == 0:
                print('  Warning: detector found no valid boxes; skipping image')
                continue
            bboxes = np.array(boxes, dtype=np.float32)
        elif args.ann_file:
            if img_name not in bbox_lookup:
                print("  Warning: No annotations for this image")
                continue
            bboxes = bbox_lookup[img_name]
        else:
            h, w = rgb_img.shape[:2]
            bboxes = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)
        
        # Run inference with manual bbox
        try:
            rgb_result = inference_topdown(model, rgb_img, bboxes=bboxes)
            ir_result = inference_topdown(model, ir_img, bboxes=bboxes)
        except Exception as e:
            print(f"  Error during inference: {e}")
            continue
        
        # Extract keypoints
        rgb_kps = extract_kps(rgb_result)
        ir_kps = extract_kps(ir_result)

        # Collect COCO-style predictions for concise AP/AR
        if coco is not None and img_name in bbox_lookup:
            img_id = None
            # Recover image_id from coco (using file_name)
            try:
                # Faster than iterating all imgs again
                img_id = coco.getImgIds(imgIds=[])
            except Exception:
                img_id = None
            # If we still don't have it, rebuild mapping quickly
            if img_id is None:
                pass  # will fallback below
            # Quick lookup via annotations already built
            # We re-compute mapping only once lazily
            # Build lightweight filename->id map if needed
        
        if coco is not None:
            # We can query by filename by scanning once (COCO API lacks direct lookup)
            # Use a cached map built earlier as `name_to_id`
            # If not available in scope (older Python), rebuild minimal on demand
            if 'name_to_id' not in locals():
                name_to_id = {}
                for _img_id in coco.getImgIds():
                    info = coco.loadImgs(_img_id)[0]
                    name_to_id[info['file_name']] = _img_id
            img_id = name_to_id.get(img_name, None)
            if img_id is not None:
                coco_preds_ir.extend(_collect_coco_results(ir_result, img_id, expected_kpt_num=effective_k))
                coco_preds_rgb.extend(_collect_coco_results(rgb_result, img_id, expected_kpt_num=effective_k))

        # Compute distance
        if rgb_kps is not None and ir_kps is not None:
            if rgb_kps.ndim == 2:
                rgb_kps = rgb_kps[np.newaxis, ...]
            if ir_kps.ndim == 2:
                ir_kps = ir_kps[np.newaxis, ...]

            num_instances = min(len(rgb_kps), len(ir_kps))
            for inst_idx in range(num_instances):
                dist = normalized_keypoint_distance(rgb_kps[inst_idx], ir_kps[inst_idx])
                per_person_distances.append(dist)
            mean_inst_dist = np.mean(per_person_distances[-num_instances:]) if num_instances > 0 else float('inf')
            distances.append(mean_inst_dist)
            print(f"  Mean normalized distance (per person): {mean_inst_dist:.4f} over {num_instances} instances")
        else:
            print(f"  Warning: Could not extract keypoints")
        
        # Visualize and save
        try:
            # Draw keypoints on RGB
            rgb_vis = rgb_img.copy()
            if rgb_kps is not None:
                for inst_idx, kp in enumerate(rgb_kps):
                    color_rng = np.random.default_rng(seed=hash((img_name, inst_idx)) & 0xFFFFFFFF)
                    color = tuple(int(c) for c in color_rng.integers(0, 255, size=3))
                    if args.ann_file and inst_idx < len(bboxes):
                        box = bboxes[inst_idx]
                        cv2.rectangle(rgb_vis, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), color, 2)
                    for x, y in kp[:, :2]:
                        cv2.circle(rgb_vis, (int(x), int(y)), 3, color, -1)
            cv2.imwrite(os.path.join(args.out_dir, 'rgb', img_name), rgb_vis)
            
            # Draw keypoints on IR
            ir_vis = ir_img.copy()
            if ir_kps is not None:
                for inst_idx, kp in enumerate(ir_kps):
                    color_rng = np.random.default_rng(seed=hash((img_name, inst_idx)) & 0xFFFFFFFF)
                    color = tuple(int(c) for c in color_rng.integers(0, 255, size=3))
                    if args.ann_file and inst_idx < len(bboxes):
                        box = bboxes[inst_idx]
                        cv2.rectangle(ir_vis, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), color, 2)
                    for x, y in kp[:, :2]:
                        cv2.circle(ir_vis, (int(x), int(y)), 3, color, -1)
            cv2.imwrite(os.path.join(args.out_dir, 'ir', img_name), ir_vis)
            
        except Exception as e:
            print(f"  Warning: Visualization error: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(distances)}")
    if distances:
        print(f"Mean image distance: {np.mean(distances):.4f}")
        print(f"Median image distance: {np.median(distances):.4f}")
        print(f"Std image distance: {np.std(distances):.4f}")
        print(f"Min image distance: {np.min(distances):.4f}")
        print(f"Max image distance: {np.max(distances):.4f}")
    if per_person_distances:
        print("\nPer-person statistics:")
        print(f"  Mean:   {np.mean(per_person_distances):.4f}")
        print(f"  Median: {np.median(per_person_distances):.4f}")
        print(f"  Std:    {np.std(per_person_distances):.4f}")
        print(f"  Min:    {np.min(per_person_distances):.4f}")
        print(f"  Max:    {np.max(per_person_distances):.4f}")
    else:
        print("No valid distance measurements obtained")
    print(f"\nVisualizations saved to:")
    print(f"  RGB: {os.path.join(args.out_dir, 'rgb')}")
    print(f"  IR:  {os.path.join(args.out_dir, 'ir')}")
    
    # Concise COCO-WholeBody metrics (Whole AP / Whole AR) for IR predictions
    if coco is not None and COCOeval is not None and len(coco_preds_ir) > 0:
        try:
            coco_dt = coco.loadRes(coco_preds_ir)
            coco_eval = COCOeval(coco, coco_dt, 'keypoints')
            if effective_k is not None:
                if effective_k == 133:
                    if WHOLEBODY_OKS_SIGMAS is not None and len(WHOLEBODY_OKS_SIGMAS) == 133:
                        coco_eval.params.kpt_oks_sigmas = np.array(WHOLEBODY_OKS_SIGMAS)
                    else:
                        coco_eval.params.kpt_oks_sigmas = np.full(133, 0.05, dtype=np.float32)
                elif effective_k == 17:
                    coco_eval.params.kpt_oks_sigmas = DEFAULT_COCO17_SIGMAS
            coco_eval.evaluate()
            coco_eval.accumulate()
            # Do not print full table; just summarize to get stats vector
            coco_eval.summarize()
            whole_ap = float(coco_eval.stats[0])
            whole_ar = float(coco_eval.stats[5])

            # Compact, one-line summary similar ao MMPose
            print("\nConcise keypoint metrics (IR predictions):")
            k = effective_k if effective_k is not None else 'N/A'
            print(f"  AP: {whole_ap:.4f} | AR: {whole_ar:.4f} | K={k}")

            # Save JSON with concise metrics
            concise = {
                'whole_ap': whole_ap,
                'whole_ar': whole_ar,
                'num_images': len(rgb_images),
                'num_predictions': len(coco_preds_ir),
            }
            with open(os.path.join(args.out_dir, 'wholebody_metrics_ir.json'), 'w') as f:
                json.dump(concise, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to compute COCO-WholeBody AP/AR: {e}")

    # Optional: concise metrics for RGB predictions on the same subset
    if coco is not None and COCOeval is not None and len(coco_preds_rgb) > 0:
        try:
            coco_dt = coco.loadRes(coco_preds_rgb)
            coco_eval = COCOeval(coco, coco_dt, 'keypoints')
            if effective_k is not None:
                if effective_k == 133:
                    if WHOLEBODY_OKS_SIGMAS is not None and len(WHOLEBODY_OKS_SIGMAS) == 133:
                        coco_eval.params.kpt_oks_sigmas = np.array(WHOLEBODY_OKS_SIGMAS)
                    else:
                        coco_eval.params.kpt_oks_sigmas = np.full(133, 0.05, dtype=np.float32)
                elif effective_k == 17:
                    coco_eval.params.kpt_oks_sigmas = DEFAULT_COCO17_SIGMAS
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            whole_ap = float(coco_eval.stats[0])
            whole_ar = float(coco_eval.stats[5])
            print("\nConcise keypoint metrics (RGB predictions):")
            k = effective_k if effective_k is not None else 'N/A'
            print(f"  AP: {whole_ap:.4f} | AR: {whole_ar:.4f} | K={k}")
            concise = {
                'whole_ap': whole_ap,
                'whole_ar': whole_ar,
                'num_images': len(rgb_images),
                'num_predictions': len(coco_preds_rgb),
            }
            with open(os.path.join(args.out_dir, 'wholebody_metrics_rgb.json'), 'w') as f:
                json.dump(concise, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to compute COCO-WholeBody AP/AR for RGB: {e}")

    print("="*60)


if __name__ == '__main__':
    main()
