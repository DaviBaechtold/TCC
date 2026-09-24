#!/usr/bin/env bash
# Retreina o lifting veicular com a perda que respeita o peso do alvo.
#
# O primeiro treino (scripts/run_lifting_veicular.sh) usou a
# MPJPEVelocityJointLoss com `use_target_weight` no padrão, desligado. Os 121
# pontos sem referência de cada janela do Drive&Act têm alvo no mesmo ponto, e a
# rede aprendeu a colapsá-los: face, mãos e pernas. A régua do Drive&Act, que
# mede só os 12 pontos corporais, não viu.
#
# Esta fila isola uma variável: mesmo conjunto, mesmo ponto de partida, mesmas
# épocas; muda só o peso na perda. E mede o modelo antigo e o novo com o mesmo
# instrumento, que agora inclui a geometria dos 133 pontos no próprio domínio.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"
LOG="$LOGS/lifting_veicular_peso.log"

# Marcador de retomada: scripts/retomar_apos_reboot.sh relança esta fila se a
# máquina cair. Só some quando a fila conclui --- sem `trap EXIT`, pelo motivo
# registrado em run_lifting_veicular.sh.
echo "run_lifting_veicular_peso.sh" > "$LOGS/fila_pendente"
agora() { date '+%H:%M:%S'; }

ANTIGO=work_dirs/lift3d_veicular/best_MPJPE_whole_epoch_4.pth
NOVO_DIR=work_dirs/lift3d_veicular_peso

# Mesma chamada para os dois modelos: é ela, e não o script, que torna os
# números comparáveis.
validar() {
    python scripts/validate_lifting_driveact.py \
        --poses ~/Downloads/extracted/openpose_3d \
        --lift-cfg configs/lift3d_veicular.py --lift-ckpt "$1" \
        --confidence normalizada \
        --max-sequences 4 --max-frames-per-sequence 120 \
        --out "$2" >> "$LOG" 2>&1
}

# Processo de computação na GPU, e não `pgrep`: o padrão casaria com o próprio
# shell que escreveu este arquivo.
ocupada() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while ocupada; do sleep 60; done
echo "[$(agora)] GPU livre"

# O modelo com defeito, medido com o instrumento novo. É a linha de base da
# comparação e o ensaio do validador: se ele falhar, falha em minutos, e não
# depois de uma hora de treino.
echo "[$(agora)] medindo o modelo com o defeito de perda"
validar "$ANTIGO" results/lifting_driveact_veicular_defeito.json \
    || { echo "[$(agora)] validação falhou antes do treino"; exit 1; }

echo "[$(agora)] treinando com a perda corrigida"
for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_veicular.py --resume >> "$LOG" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t "$NOVO_DIR"/best_MPJPE_whole_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"
validar "$MELHOR" results/lifting_driveact_veicular_peso.json

# O `best` é escolhido pelo MPJPE no H3WB, que não mede o domínio veicular.
ULTIMA=$(ls -t "$NOVO_DIR"/epoch_*.pth 2>/dev/null | head -1)
if [ -n "$ULTIMA" ] && [ "$ULTIMA" != "$MELHOR" ]; then
    validar "$ULTIMA" results/lifting_driveact_veicular_peso_ultima.json
fi

# Os dois modelos ao vivo, sob as condições de hoje. A medida ao vivo antiga do
# modelo com defeito foi feita com o RTMDet como detector; remedi-la aqui faz a
# comparação isolar a perda, e não a perda somada à troca de detector.
ao_vivo() {
    python scripts/measure_live_quality.py --tag "$1" --distancia 1.11 \
        --video work_dirs/panel/rec_20260921_224039.mp4 \
        --lift-cfg configs/lift3d_veicular.py --lift-ckpt "$2" >> "$LOG" 2>&1
}
echo "[$(agora)] qualidade ao vivo na gravação com tabuleiro, os dois modelos"
ao_vivo veicular_defeito_ao_vivo "$ANTIGO"
ao_vivo veicular_peso_ao_vivo "$MELHOR"

echo "[$(agora)] concluído"
rm -f "$LOGS/fila_pendente"
