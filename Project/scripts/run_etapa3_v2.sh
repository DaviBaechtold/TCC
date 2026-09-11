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

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_etapa3_v2.sh" > "$LOGS/fila_pendente"
trap 'rm -f "$LOGS/fila_pendente"' EXIT
agora() { date '+%H:%M:%S'; }

# Espera a GPU esvaziar em vez de procurar o processo pelo nome.
#
# `pgrep -f <padrão>` casa com qualquer linha de comando que contenha o padrão,
# **inclusive a do shell que escreveu este arquivo**: o heredoc que o criou tem
# o padrão dentro dele. A fila ficou 2h10 esperando o próprio criador terminar,
# que é algo que não acontece enquanto ela roda. Já tinha mordido antes com
# `pkill -f`, matando o shell que emitia o comando.
#
# A memória da GPU não mente e não se auto-referencia.
ocupada() {
    local usada
    usada=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${usada:-0}" -gt 1500 ]
}
while ocupada; do sleep 60; done
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
