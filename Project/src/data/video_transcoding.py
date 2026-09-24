"""Recodifica um vídeo como uma câmera de rede o transmitiria.

Camada Model. Existe para medir o que um stream de vídeo faz com o estimador:
testar o sistema fora de casa exige mandar a imagem da câmera do carro para o PC
por Wi-Fi ou 4G, e a banda disponível decide a compressão. A pergunta é quanto
de compressão o infravermelho aguenta, e ela se responde recodificando os vídeos
do Drive&Act e medindo o erro nos mesmos quadros anotados.

Os parâmetros imitam uma câmera de rede em transmissão ao vivo, e não um arquivo
para arquivamento: taxa constante, porque o canal tem banda fixa; um quadro-chave
a cada dois segundos, a faixa usual de câmeras IP; e nenhum quadro B, porque ele
depende de quadros futuros e acrescenta atraso, o que um stream de baixa
latência não admite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Dois segundos a 30 FPS.
STREAM_GOP_FRAMES = 60

# O codificador de hardware da GPU. Cameras e celulares também codificam em
# hardware, e ele recodifica 25 minutos de vídeo em poucos minutos, contra horas
# do libx264 na CPU.
STREAM_ENCODER = 'h264_nvenc'


def transcode_for_stream(source: Path, destination: Path,
                         bitrate_kbps: int) -> Path:
    """Recodifica `source` em H.264 a `bitrate_kbps` constantes.

    Returns:
        O caminho do vídeo recodificado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = f'{bitrate_kbps}k'
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(source),
         '-c:v', STREAM_ENCODER, '-rc', 'cbr', '-b:v', rate,
         '-maxrate', rate, '-bufsize', rate,
         '-g', str(STREAM_GOP_FRAMES), '-bf', '0', '-an', str(destination)],
        check=True)
    return destination
