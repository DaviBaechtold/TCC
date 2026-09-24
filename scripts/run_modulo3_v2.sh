#!/usr/bin/env bash
# Treina o lifting com o estimador 2D simulado e mede se o domínio veicular
# melhora. Compara contra dois pontos já medidos: 82,88mm do modelo base e
# 108,37mm da primeira tentativa, que piorou.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_modulo3_v2.sh" > "$LOGS/fila_pendente"
# O marcador só some quando a fila **conclui**, e é por isso que não há `trap
# EXIT` aqui. O trap dispara também quando o shell é morto no desligamento da
# máquina — exatamente o momento em que o marcador precisa sobreviver. Foi o que
# aconteceu em 12/09: a máquina reiniciou, o trap apagou o marcador antes de
# morrer, e a retomada automática não achou o que religar.
agora() { date '+%H:%M:%S'; }

for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_robusto.py --resume \
        >> "$LOGS/lift_v2.log" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t work_dirs/lift3d_robusto_v2/best_MPJPE_whole_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

echo "[$(agora)] validando no domínio veicular"
python scripts/validate_lifting_driveact.py \
    --poses ~/Downloads/extracted/openpose_3d --lift-ckpt "$MELHOR" \
    --confidence normalizada \
    --max-sequences 4 --max-frames-per-sequence 120 \
    --out results/lifting_driveact_v2.json >> "$LOGS/lift_v2.log" 2>&1
echo "[$(agora)] concluído"

rm -f "$LOGS/fila_pendente"
