"""Recodifica um vídeo como uma câmera de monitoramento o codificaria.

Camada Model. Existe para medir o que a compressão do vídeo faz com o estimador:
toda câmera entrega H.264, inclusive a infravermelha própria pendente para a
validação, e a pergunta é quanta compressão o infravermelho aguenta. Ela se
responde recodificando os vídeos do Drive&Act e medindo o erro nos mesmos quadros
anotados.

Os parâmetros imitam o codificador de uma câmera, e não um arquivo para
arquivamento: taxa constante, que é como esses aparelhos a configuram; um
quadro-chave a cada dois segundos, a faixa usual; e nenhum quadro B, que as
câmeras evitam porque dependem de quadros futuros e acrescentam atraso.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Dois segundos a 30 FPS.
CAMERA_GOP_FRAMES = 60

# O codificador de hardware da GPU. Câmeras e celulares também codificam em
# hardware, e ele recodifica 25 minutos de vídeo em poucos minutos, contra horas
# do libx264 na CPU.
CAMERA_ENCODER = 'h264_nvenc'


def transcode_like_camera(source: Path, destination: Path,
                          bitrate_kbps: int) -> Path:
    """Recodifica `source` em H.264 a `bitrate_kbps` constantes.

    Returns:
        O caminho do vídeo recodificado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = f'{bitrate_kbps}k'
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(source),
         '-c:v', CAMERA_ENCODER, '-rc', 'cbr', '-b:v', rate,
         '-maxrate', rate, '-bufsize', rate,
         '-g', str(CAMERA_GOP_FRAMES), '-bf', '0', '-an', str(destination)],
        check=True)
    return destination
