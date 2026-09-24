#!/usr/bin/env bash
# QP5 (latência do lifting contra profundidade e precisão) e QP4 (ganho da
# janela temporal contra quadro único).
#
# Roda depois do retreino veicular. Espera o marcador daquela fila sumir, e não a
# GPU esvaziar: entre os passos de uma fila a GPU fica livre por instantes, e as
# duas se atropelariam.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"
LOG="$LOGS/qp4_qp5.log"
agora() { date '+%H:%M:%S'; }

while [ "$(cat "$LOGS/fila_pendente" 2>/dev/null)" = "run_lifting_veicular_peso.sh" ]; do
    sleep 60
done
echo "run_qp4_qp5.sh" > "$LOGS/fila_pendente"
ocupada() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while ocupada; do sleep 30; done
echo "[$(agora)] GPU livre"

# QP5 primeiro: é medição de latência, e precisa da GPU só para ela.
echo "[$(agora)] QP5: latência do lifting"
python scripts/benchmark_lifting.py \
    --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v3.py \
    --lift-ckpt work_dirs/lift3d_robusto_v3/best_MPJPE_whole_epoch_12.pth \
    >> "$LOG" 2>&1 || echo "[$(agora)] QP5 falhou; segue para a QP4"

# QP4: o mesmo treino base, com janela de um quadro.
echo "[$(agora)] QP4: treinando o lifting de quadro único"
for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_1frm.py --resume >> "$LOG" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

echo "[$(agora)] concluído"
rm -f "$LOGS/fila_pendente"
