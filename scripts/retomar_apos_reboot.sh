#!/usr/bin/env bash
# Retoma a fila de treinos que estava rodando quando a máquina caiu.
#
# Existe porque esta máquina cai. O diagnóstico está no CLAUDE.md: 129
# desligamentos inesperados em 503 ciclos de energia, evidência apontando para
# entrega de energia, e travamentos que aconteceram inclusive com o computador
# ocioso. O treino em si já sabe retomar — o `CheckpointHook` grava a cada
# época e `--resume` encontra o mais recente. O que não sobrevivia ao reboot era
# a **fila**, que morria junto com a sessão.
#
# Instalar com:  crontab -l | { cat; echo "@reboot $PWD/scripts/retomar_apos_reboot.sh"; } | crontab -
set -u
cd "$(dirname "$0")/.."

LOGS=work_dirs/logs
mkdir -p "$LOGS"
PENDENTE="$LOGS/fila_pendente"

# A fila em execução registra seu nome aqui e apaga ao terminar. Sem o arquivo,
# não havia nada rodando e não há o que retomar.
[ -f "$PENDENTE" ] || exit 0
FILA=$(cat "$PENDENTE")
[ -x "scripts/$FILA" ] || exit 0

# Espera a máquina assentar: driver da GPU, montagem de disco, rede.
sleep 90

echo "[$(date '+%F %H:%M:%S')] retomando $FILA após reinício" >> "$LOGS/reinicios.log"
setsid "./scripts/$FILA" >> "$LOGS/${FILA%.sh}_fila.log" 2>&1 < /dev/null &

# A vigia sobe junto, senão a máquina volta sem quem a proteja da próxima queda
# --- e a queda que ela cobre, a GPU saindo do barramento com o sistema de pé,
# não dispara o watchdog de hardware porque o kernel continua vivo.
setsid ./scripts/vigia_hardware.sh 300 >> "$LOGS/vigia.log" 2>&1 < /dev/null &
