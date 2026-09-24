#!/usr/bin/env bash
# Controle da QP4: quadro único com o lote de 4 do treino base.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; LOG="$LOGS/qp4_controle.log"
echo "run_qp4_controle.sh" > "$LOGS/fila_pendente"
agora() { date '+%H:%M:%S'; }
ocupada() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while ocupada; do sleep 30; done
echo "[$(agora)] treinando o quadro único com lote 4"
for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_1frm_lote4.py --resume >> "$LOG" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done
echo "[$(agora)] concluído"
rm -f "$LOGS/fila_pendente"
