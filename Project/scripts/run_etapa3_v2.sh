#!/usr/bin/env bash
# Etapa 3 v2: supervisiona as juntas fora de quadro e mede se a confiança nelas
# volta a separar do que é observado.
#
# Espera a fila do lifting terminar. Medir sob disputa de GPU dá um terço do
# valor real e rouba tempo do treino que se quer avaliar.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"
agora() { date '+%H:%M:%S'; }

while pgrep -f "run_modulo3_v2.sh" > /dev/null \
   || pgrep -f "lift3d_dstformer_h3wb_robusto" > /dev/null; do sleep 60; done
echo "[$(agora)] GPU livre; treinando a Etapa 3 v2"

for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/rtmw_x_driveact_ft_v2.py --resume \
        >> "$LOGS/etapa3_v2.log" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t work_dirs/rtmw_x_driveact_ft_v2/best_torso_px_mean_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

echo "[$(agora)] avaliando no Drive&Act"
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_driveact_eval.py \
    --ckpt "$MELHOR" --data-root data/processed/driveact/ --img-prefix val/ \
    --ann-file driveact_midlevel.chunks_90.split_0.val.json \
    --tag rtmwx_etapa3v2_driveact --batch-size 8 >> "$LOGS/etapa3_v2.log" 2>&1

FUNDIDO="${MELHOR%.pth}_merged.pth"
echo "[$(agora)] medindo a separação de confiança e o efeito no lifting"
python scripts/measure_confidence_separation.py --ckpt "$FUNDIDO" \
    >> "$LOGS/etapa3_v2.log" 2>&1
python scripts/validate_lifting_driveact.py \
    --poses ~/Downloads/extracted/openpose_3d --pose-ckpt "$FUNDIDO" \
    --max-sequences 4 --max-frames-per-sequence 120 \
    --out results/lifting_driveact_etapa3v2.json >> "$LOGS/etapa3_v2.log" 2>&1
echo "[$(agora)] concluído"
