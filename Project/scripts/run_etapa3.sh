#!/usr/bin/env bash
# Etapa 3 — adaptação ao domínio veicular real, com retomada automática.
#
# Separado de run_overnight.sh porque depende do resultado da Etapa 2, que
# aquele script produz. Encadeá-los num arquivo só obrigaria a reexecutar a
# Etapa 2 para retomar esta.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

LOGS=work_dirs/logs
mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_etapa3.sh" > "$LOGS/fila_pendente"
# O marcador só some quando a fila **conclui**, e é por isso que não há `trap
# EXIT` aqui. O trap dispara também quando o shell é morto no desligamento da
# máquina — exatamente o momento em que o marcador precisa sobreviver. Foi o que
# aconteceu em 12/09: a máquina reiniciou, o trap apagou o marcador antes de
# morrer, e a retomada automática não achou o que religar.
agora() { date '+%H:%M:%S'; }

# Retomar em vez de recomeçar: a máquina tem histórico de travar sozinha, e
# perder quatro horas de treino por um reboot seria evitável.
for tentativa in 1 2 3; do
    echo "[$(agora)] etapa3: tentativa $tentativa"
    python scripts/train_wholebody.py \
        --config configs/rtmw_x_driveact_ft.py --resume \
        >> "$LOGS/etapa3.log" 2>&1 && {
        echo "[$(agora)] etapa3: OK"
        exit 0
    }
    echo "[$(agora)] etapa3: falhou (tentativa $tentativa)"
    sleep 30
done
echo "[$(agora)] etapa3: DESISTINDO após 3 tentativas"
exit 1

rm -f "$LOGS/fila_pendente"
