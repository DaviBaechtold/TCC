#!/usr/bin/env python
"""Converte o Drive&Act para o formato COCO-WholeBody.

Controller: lê argumentos, orquestra as funções de `src/data/driveact.py` e
reporta o progresso. Toda a regra de conversão vive no módulo, não aqui.

Produz, para um split, um JSON de anotações COCO e os frames correspondentes
extraídos dos vídeos NIR.

Exemplo:
    python scripts/convert_driveact.py \\
        --videos ~/Downloads/extracted/inner_mirror \\
        --poses ~/Downloads/extracted/openpose_3d \\
        --splits ~/Downloads/extracted/activities_3s/inner_mirror \\
        --split-name midlevel.chunks_90.split_0 \\
        --output data/processed/driveact \\
        --subset train --frame-stride 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.driveact import (
    MIN_VISIBLE_KEYPOINTS,
    NUM_MAPPED_KEYPOINTS,
    CameraCalibration,
    activity_by_frame,
    bounding_box_from_keypoints,
    extract_frames,
    read_pose_csv,
    read_split,
    to_coco_keypoints,
)

# Nomes das 23 categorias mapeadas, na ordem do COCO-WholeBody. As 110 restantes
# (face e mãos) existem no vetor de keypoints mas nunca são anotadas aqui.
CATEGORY = {
    'id': 1,
    'name': 'person',
    'supercategory': 'person',
    'keypoints': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel',
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', type=Path, required=True,
                        help='Diretório inner_mirror/ com os .mp4 e .calibration.json')
    parser.add_argument('--poses', type=Path, required=True,
                        help='Diretório openpose_3d/ com os .openpose.3d.csv')
    parser.add_argument('--splits', type=Path, required=True,
                        help='Diretório activities_3s/inner_mirror/')
    parser.add_argument('--split-name', default='midlevel.chunks_90.split_0',
                        help='Prefixo do arquivo de split')
    parser.add_argument('--subset', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-stride', type=int, default=5,
                        help='Amostra 1 frame a cada N. Frames consecutivos a 30 FPS '
                             'são quase idênticos e inflam o dataset sem agregar pose nova')
    parser.add_argument('--skip-frame-extraction', action='store_true',
                        help='Gera apenas o JSON, útil para inspecionar antes de gastar disco')
    return parser.parse_args()


def main():
    args = parse_args()

    split_file = args.splits / f'{args.split_name}.{args.subset}.csv'
    if not split_file.exists():
        raise SystemExit(f'Split não encontrado: {split_file}')

    segments = read_split(split_file)
    files = sorted({segment.file_id for segment in segments})
    participants = sorted({segment.participant_id for segment in segments})
    print(f'{split_file.name}: {len(segments)} segmentos, '
          f'{len(files)} gravações, participantes {participants}')

    image_dir = args.output / args.subset
    images, annotations = [], []
    activity_counts = Counter()
    next_image_id = next_annotation_id = 1

    for file_id in files:
        pose_csv = args.poses / f'{file_id}.openpose.3d.csv'
        calib_json = args.videos / f'{file_id}.calibration.json'
        video = args.videos / f'{file_id}.mp4'

        required = [pose_csv, calib_json]
        if not args.skip_frame_extraction:
            required.append(video)
        missing = [p for p in required if not p.exists()]
        if missing:
            print(f'  {file_id}: ausente {[p.name for p in missing]}, pulando')
            continue

        calibration = CameraCalibration.from_json(calib_json)
        labels = activity_by_frame([s for s in segments if s.file_id == file_id])

        pending = []
        for pose in read_pose_csv(pose_csv):
            if pose.frame_id % args.frame_stride:
                continue
            if pose.num_visible < MIN_VISIBLE_KEYPOINTS:
                continue
            activity = labels.get(pose.frame_id)
            if activity is None:
                continue  # frame fora de qualquer segmento rotulado

            visible = pose.confidence > 0
            points_2d = np.zeros((NUM_MAPPED_KEYPOINTS, 2), dtype=np.float32)
            points_2d[visible] = calibration.project(pose.points_3d[visible])

            # Uma junta atrás da câmera ou fora do quadro projeta para uma
            # posição sem sentido; tratá-la como não anotada é mais honesto do
            # que gravar a coordenada extrapolada.
            width, height = calibration.image_size
            inside = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
                      (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height))
            confidence = np.where(visible & inside, pose.confidence, 0.0)
            if (confidence > 0).sum() < MIN_VISIBLE_KEYPOINTS:
                continue

            pending.append((pose.frame_id, points_2d, confidence, activity))

        if not pending:
            print(f'  {file_id}: nenhum frame utilizável')
            continue

        prefix = file_id.replace('/', '_')
        frame_ids = [frame_id for frame_id, *_ in pending]
        if args.skip_frame_extraction:
            filenames = {i: f'{prefix}_{i:06d}.jpg' for i in frame_ids}
        else:
            filenames = extract_frames(video, frame_ids, image_dir, prefix)

        kept = 0
        for frame_id, points_2d, confidence, activity in pending:
            filename = filenames.get(frame_id)
            if filename is None:
                continue  # o vídeo acabou antes deste frame

            keypoints, num_visible = to_coco_keypoints(points_2d, confidence)
            x, y, w, h = bounding_box_from_keypoints(
                points_2d, confidence > 0, calibration.image_size)
            if w <= 0 or h <= 0:
                continue

            width, height = calibration.image_size
            images.append({
                'id': next_image_id, 'file_name': filename,
                'width': width, 'height': height,
                # Metadados fora do padrão COCO, usados para estratificar as
                # métricas por atividade e por participante na avaliação.
                'activity': activity, 'file_id': file_id, 'frame_id': frame_id,
            })
            annotations.append({
                'id': next_annotation_id, 'image_id': next_image_id,
                'category_id': 1, 'iscrowd': 0,
                'bbox': [x, y, w, h], 'area': w * h,
                'num_keypoints': num_visible, 'keypoints': keypoints,
            })
            activity_counts[activity] += 1
            next_image_id += 1
            next_annotation_id += 1
            kept += 1

        print(f'  {file_id}: {kept} frames')

    args.output.mkdir(parents=True, exist_ok=True)
    out_json = args.output / f'driveact_{args.split_name}.{args.subset}.json'
    out_json.write_text(json.dumps({
        'images': images, 'annotations': annotations, 'categories': [CATEGORY],
    }))

    print(f'\n{len(images)} imagens, {len(annotations)} anotações -> {out_json}')
    print('\nDistribuição por atividade:')
    for activity, count in activity_counts.most_common(12):
        print(f'  {count:7d}  {activity}')


if __name__ == '__main__':
    main()
