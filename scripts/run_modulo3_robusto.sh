#!/usr/bin/env bash
# Treina o Módulo 3 para tolerar entrada incompleta, e mede se isso fecha o
# salto de domínio.
#
# Espera a fila anterior terminar em vez de disputar a GPU. Um script separado,
# e não mais um passo na outra fila, porque editar um script de shell em
# execução corrompe o que ele lê em seguida — o bash lê por offset de bytes.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

LOGS=work_dirs/logs
mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_modulo3_robusto.sh" > "$LOGS/fila_pendente"
# O marcador só some quando a fila **conclui**, e é por isso que não há `trap
# EXIT` aqui. O trap dispara também quando o shell é morto no desligamento da
# máquina — exatamente o momento em que o marcador precisa sobreviver. Foi o que
# aconteceu em 12/09: a máquina reiniciou, o trap apagou o marcador antes de
# morrer, e a retomada automática não achou o que religar.
agora() { date '+%H:%M:%S'; }

while pgrep -f "run_pos_etapa3.sh" > /dev/null \
   || pgrep -f "configs/rtmw_x_driveact_ft.py" > /dev/null; do sleep 60; done
echo "[$(agora)] GPU livre; treinando o lifting robusto"

for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_robusto.py --resume \
        >> "$LOGS/lift_robusto.log" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t work_dirs/lift3d_robusto/best_MPJPE_whole_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint do lifting robusto"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

# A pergunta que fecha o ciclo: o erro no habitáculo cai dos 82,88mm medidos?
FUNDIDO=$(ls -t work_dirs/rtmw_x_driveact_ft/best_torso_px_mean_*_merged.pth 2>/dev/null | head -1)
echo "[$(agora)] validando no domínio veicular"
python scripts/validate_lifting_driveact.py \
    --poses ~/Downloads/extracted/openpose_3d \
    ${FUNDIDO:+--pose-ckpt "$FUNDIDO"} \
    --lift-ckpt "$MELHOR" \
    --max-sequences 4 --max-frames-per-sequence 120 \
    --out results/lifting_driveact_robusto.json >> "$LOGS/lift_robusto.log" 2>&1

echo "[$(agora)] fila do Módulo 3 robusto concluída"

rm -f "$LOGS/fila_pendente"
