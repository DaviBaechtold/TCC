#!/bin/bash
# Fila de treinos longos, resiliente a quedas da máquina.
#
# Cada etapa retoma do próprio checkpoint, de modo que relançar este script
# depois de um travamento continua de onde parou em vez de recomeçar. O estado
# fica num arquivo, não em memória, pela mesma razão.
#
# Uso:  nohup ./scripts/run_overnight.sh > work_dirs/logs/fila.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

LOGS=work_dirs/logs
STATE=$LOGS/fila.estado
mkdir -p "$LOGS"
touch "$STATE"

concluida() { grep -qx "$1" "$STATE"; }
marcar()    { echo "$1" >> "$STATE"; }
agora()     { date '+%H:%M:%S'; }

# Executa uma etapa com tentativas. Treinos retomam do checkpoint a cada
# tentativa, então uma queda custa no máximo a época corrente.
executar() {
    local nome="$1"; shift
    if concluida "$nome"; then
        echo "[$(agora)] $nome: já concluída, pulando"
        return 0
    fi
    for tentativa in 1 2 3; do
        echo "[$(agora)] $nome: tentativa $tentativa"
        if "$@"; then
            marcar "$nome"
            echo "[$(agora)] $nome: OK"
            return 0
        fi
        echo "[$(agora)] $nome: falhou (tentativa $tentativa)"
        sleep 30
    done
    echo "[$(agora)] $nome: DESISTINDO após 3 tentativas"
    return 1
}

lora_treino() {
    python scripts/train_wholebody.py \
        --config configs/rtmw_x_wholebody_gray_lora.py --resume \
        >> "$LOGS/lora.log" 2>&1
}

lora_avaliacao() {
    local ckpt
    ckpt=$(ls -t work_dirs/rtmw_x_gray_lora/best_*.pth 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "sem checkpoint do LoRA"; return 1; }
    python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
        --ckpt "$ckpt" --data-root data/processed/grayscale/ \
        --tag rtmwx_lora_gray --batch-size 8 >> "$LOGS/aval_lora.log" 2>&1
}

lift_treino() {
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_16frm.py --resume \
        >> "$LOGS/lift.log" 2>&1
}

echo "================ fila iniciada $(date) ================"
executar "lora_treino"     lora_treino
executar "lora_avaliacao"  lora_avaliacao
executar "lift_treino"     lift_treino
echo "================ fila encerrada $(date) ================"
