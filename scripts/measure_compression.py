#!/usr/bin/env python
"""Mede quanto a compressão do vídeo degrada o estimador 2D.

Controller. Toda câmera entrega o vídeo comprimido em H.264, inclusive a
infravermelha própria pendente para a validação. O infravermelho é escuro e
ruidoso, o tipo de imagem que o H.264 borra primeiro, e a pergunta é a partir de
que taxa o sistema deixa de medir o mesmo que mede no conjunto.

Os vídeos do Drive&Act já chegam comprimidos, entre 0,49 e 0,54 Mbps em
1280x1024, e são o teto de qualidade desta medição: ela não diz nada sobre taxas
maiores que a do original, só sobre o que se perde abaixo dele. A condição de
2 Mbps é o controle --- recodificar sem apertar a taxa não deveria mudar o erro.

Os quatro vídeos da partição de validação são recodificados por inteiro, para
que o controle de taxa do codificador opere como numa gravação longa, e os mesmos
quadros anotados são extraídos e medidos com o estimador de operação.

    python scripts/measure_compression.py
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detector_config import DEFAULT_DETECTOR, DETECTOR_CHOICES

FRAMES = Path('data/processed/driveact/val')
ANNOTATIONS = Path('data/processed/driveact/'
                   'driveact_midlevel.chunks_90.split_0.val.json')
VIDEOS = Path.home() / 'Downloads/extracted/inner_mirror'
WORK_DIR = Path('work_dirs/compressao')
ORIGINAL = 'original'

# Do controle até quatro vezes abaixo da taxa do original.
DEFAULT_BITRATES_KBPS = (2000, 1000, 500, 250, 125)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bitrates', type=int, nargs='+',
                   default=list(DEFAULT_BITRATES_KBPS),
                   help='Taxas em kbps')
    p.add_argument('--max-images', type=int, default=1500,
                   help='Mesmo teto da comparação de detectores, para que os '
                        'números se comparem')
    p.add_argument('--detector', default=DEFAULT_DETECTOR,
                   choices=DETECTOR_CHOICES)
    p.add_argument('--videos', type=Path, default=VIDEOS)
    p.add_argument('--frames', type=Path, default=FRAMES)
    p.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path,
                   default=Path('results/compressao_video.json'))
    return p.parse_args()


def measured_bitrate_kbps(video: Path) -> float:
    """A taxa que o arquivo de fato tem, e não a pedida ao codificador."""
    duration = float(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(video)],
        check=True, capture_output=True, text=True).stdout)
    return video.stat().st_size * 8 / duration / 1000


def main():
    args = parse_args()

    import cv2

    from src.data.driveact import annotated_pairs, extract_frames
    from src.data.video_transcoding import transcode_like_camera
    from src.evaluation.normalized_keypoint_error import instance_error
    from src.models import operating_config
    from src.models.pose_pipeline import (FullBodyPosePipeline,
                                          build_person_detector)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = operating_config.POSE_CHECKPOINT_BY_MOUNTING['retrovisor']
    pipeline = FullBodyPosePipeline(
        operating_config.config_without_flip_test(operating_config.POSE_CONFIG,
                                                  WORK_DIR),
        checkpoint, args.device,
        build_person_detector(args.detector, args.device))

    pares = annotated_pairs(json.loads(args.annotations.read_text()),
                            args.max_images)
    quadros_por_video = defaultdict(list)
    for imagem, _ in pares:
        quadros_por_video[imagem['file_id']].append(imagem['frame_id'])
    print(f'{len(pares)} quadros anotados em {len(quadros_por_video)} vídeos')

    def medir(pasta: Path) -> dict:
        erros, sem_deteccao = [], 0
        for imagem, anotacao in pares:
            quadro = cv2.imread(str(pasta / imagem['file_name']))
            if quadro is None:
                continue
            resultado = pipeline(quadro)
            if resultado.num_people == 0:
                sem_deteccao += 1
                continue
            referencia = np.asarray(anotacao['keypoints'],
                                    dtype=np.float32).reshape(-1, 3)
            medido = instance_error(resultado.keypoints[0], referencia)
            if medido is not None:
                erros.append(medido[0])
        erros = np.asarray(erros)
        return {'erro_medio': round(float(erros.mean()), 4),
                'erro_mediano': round(float(np.median(erros)), 4),
                'erro_p90': round(float(np.percentile(erros, 90)), 4),
                'quadros_medidos': int(len(erros)),
                'quadros_sem_deteccao': sem_deteccao}

    resultados = {}
    fontes = {v: args.videos / f'{v}.mp4' for v in quadros_por_video}
    resultados[ORIGINAL] = medir(args.frames)
    resultados[ORIGINAL]['taxa_real_kbps'] = round(float(np.mean(
        [measured_bitrate_kbps(f) for f in fontes.values()])), 1)
    print(f'  original: {resultados[ORIGINAL]}')

    for taxa in args.bitrates:
        pasta = WORK_DIR / f'{taxa}kbps'
        taxas_reais = []
        for video, quadros in quadros_por_video.items():
            recodificado = transcode_like_camera(fontes[video],
                                                 pasta / f'{video}.mp4', taxa)
            taxas_reais.append(measured_bitrate_kbps(recodificado))
            extract_frames(recodificado, quadros, pasta,
                           filename_prefix=video.replace('/', '_'))
            # Os quadros extraídos bastam, e quatro vídeos por taxa ocupariam
            # gigabytes sem uso.
            recodificado.unlink()
        resultados[f'{taxa}kbps'] = medir(pasta)
        resultados[f'{taxa}kbps']['taxa_real_kbps'] = round(
            float(np.mean(taxas_reais)), 1)
        print(f'  {taxa} kbps: {resultados[f"{taxa}kbps"]}')

    base = resultados[ORIGINAL]['erro_medio']
    for condicao in resultados.values():
        condicao['degradacao_pct'] = round(
            100 * (condicao['erro_medio'] - base) / base, 1)

    relatorio = {
        'conjunto': 'Drive&Act val, split_0, vp14 e vp15, amostra espaçada',
        'estimador': Path(checkpoint).name,
        'detector': args.detector,
        'metrica': 'erro de keypoint corporal normalizado por tronco',
        'resolucao': '1280x1024, 30 FPS',
        'codificador': ('h264_nvenc, taxa constante, GOP de 60 quadros, '
                        'sem quadros B'),
        'ressalva': ('os vídeos originais já estão comprimidos; a medição só '
                     'fala de taxas abaixo da original'),
        'por_taxa': resultados,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f'gravado em {args.out}')


if __name__ == '__main__':
    main()
