#!/usr/bin/env bash
# Medições que só fazem sentido com a Etapa 3 concluída e a GPU livre.
#
# Espera o treino terminar em vez de rodar em paralelo: medir latência sob
# disputa de GPU dá cerca de um terço do valor real, e medir acurácia rouba
# tempo do treino que se quer avaliar.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

LOGS=work_dirs/logs
mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_pos_etapa3.sh" > "$LOGS/fila_pendente"
trap 'rm -f "$LOGS/fila_pendente"' EXIT
agora() { date '+%H:%M:%S'; }
IMAGEM=data/raw/val2017/000000000139.jpg

while pgrep -f "configs/rtmw_x_driveact_ft.py" > /dev/null; do sleep 60; done
echo "[$(agora)] treino concluído; GPU livre"

MELHOR=$(ls -t work_dirs/rtmw_x_driveact_ft/best_torso_px_mean_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "sem checkpoint da Etapa 3"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

echo "[$(agora)] 1/4 avaliação final no Drive&Act, com percentil 90"
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_driveact_eval.py \
    --ckpt "$MELHOR" --data-root data/processed/driveact/ --img-prefix val/ \
    --ann-file driveact_midlevel.chunks_90.split_0.val.json \
    --tag rtmwx_etapa3_driveact --batch-size 8 >> "$LOGS/pos_etapa3.log" 2>&1

# O checkpoint fundido que a avaliação acima produz é o modelo final do
# Módulo 2, e é ele que as medições seguintes usam.
FUNDIDO="${MELHOR%.pth}_merged.pth"

echo "[$(agora)] 2/4 throughput, quatro configurações de otimização"
for flags in "" "--channels-last" "--compile" "--channels-last --compile"; do
    tag="etapa3$(echo "$flags" | tr -d ' -' | tr 'A-Z' 'a-z')"
    python scripts/benchmark_throughput.py \
        --cfg configs/eval/rtmw_x_wholebody_eval.py --ckpt "$FUNDIDO" \
        --tag "${tag:-etapa3_base}" --image "$IMAGEM" $flags \
        >> "$LOGS/pos_etapa3.log" 2>&1
done

echo "[$(agora)] 3/4 QP1: as quatro configurações de detecção"
python scripts/compare_detectors.py --pose-ckpt "$FUNDIDO" --max-frames 1500 \
    >> "$LOGS/pos_etapa3.log" 2>&1

echo "[$(agora)] 4/4 lifting no domínio veicular, com o estimador 2D adaptado"
python scripts/validate_lifting_driveact.py \
    --poses ~/Downloads/extracted/openpose_3d --pose-ckpt "$FUNDIDO" \
    --max-sequences 4 --max-frames-per-sequence 120 \
    --out results/lifting_driveact_etapa3.json >> "$LOGS/pos_etapa3.log" 2>&1

echo "[$(agora)] fila pós-Etapa 3 concluída"
