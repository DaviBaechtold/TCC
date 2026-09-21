#!/usr/bin/env bash
# Etapa 3 com ensaio: adapta o corpo ao infravermelho sem esquecer face e mãos.
#
# O checkpoint da v2 mede 0,2330 de whole-body AP no COCO em cinza contra 0,6931
# do modelo de onde partiu, com a face errando seis vezes mais. Aqui o lote
# passa a intercalar o Drive&Act com o subconjunto do COCO que anota face e
# mãos, e a fila mede as duas coisas ao final: o erro corporal no domínio de
# aplicação, que é o que a etapa se propõe a reduzir, e o erro por região no
# COCO, que é o que ela não pode estragar.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"

echo "run_etapa3_ensaio.sh" > "$LOGS/fila_pendente"
# Sem `trap EXIT`: ele dispararia também no desligamento da máquina, apagando o
# marcador exatamente quando a retomada automática precisa dele.
agora() { date '+%H:%M:%S'; }

# Espera a GPU esvaziar pela lista de processos de computação, e não por
# `pgrep -f`, que casa com a própria linha de comando que criou este arquivo e
# já travou uma fila por 2h10.
ocupada() {
    local processos
    processos=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [ -n "$processos" ]
}
while ocupada; do sleep 60; done
echo "[$(agora)] GPU livre; treinando a Etapa 3 com ensaio"

for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/rtmw_x_driveact_ensaio.py --resume \
        >> "$LOGS/etapa3_ensaio.log" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t work_dirs/rtmw_x_driveact_ensaio/best_torso_px_mean_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

echo "[$(agora)] erro corporal no Drive&Act"
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_driveact_eval.py \
    --ckpt "$MELHOR" --data-root data/processed/driveact/ --img-prefix val/ \
    --ann-file driveact_midlevel.chunks_90.split_0.val.json \
    --tag rtmwx_ensaio_driveact --batch-size 8 >> "$LOGS/etapa3_ensaio.log" 2>&1

FUNDIDO="${MELHOR%.pth}_merged.pth"

echo "[$(agora)] whole-body AP e erro por região no COCO em cinza"
python scripts/eval_checkpoint.py --cfg configs/eval/rtmw_x_wholebody_eval.py \
    --ckpt "$FUNDIDO" --data-root data/processed/grayscale/ \
    --tag rtmwx_ensaio_gray >> "$LOGS/etapa3_ensaio.log" 2>&1
python scripts/measure_region_error.py --tag ensaio --ckpt "$FUNDIDO" \
    >> "$LOGS/etapa3_ensaio.log" 2>&1

echo "[$(agora)] separação de confiança nas juntas fora de quadro"
# Com --out explícito: o padrão do script é o arquivo que o documento cita com
# os números da v2, e a fila o sobrescreveu na primeira execução.
python scripts/measure_confidence_separation.py --ckpt "$FUNDIDO" \
    --out results/separacao_confianca_ensaio.json \
    >> "$LOGS/etapa3_ensaio.log" 2>&1
echo "[$(agora)] concluído"

rm -f "$LOGS/fila_pendente"
