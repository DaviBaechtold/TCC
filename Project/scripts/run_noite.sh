#!/usr/bin/env bash
# Encadeia as duas filas restantes da noite, uma depois da outra.
#
# Um encadeador em vez de cada fila esperando a GPU por conta própria: duas
# filas esperando a mesma condição disparam juntas quando ela se cumpre, e
# passam a disputar a GPU. Aconteceu hoje.
set -u
cd "$(dirname "$0")/.."
LOGS=work_dirs/logs; mkdir -p "$LOGS"
agora() { date '+%H:%M:%S'; }

# Espera a fila da Etapa 3 v2, que já está rodando.
while pgrep -x python > /dev/null && \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '{exit !($1 > 1500)}'; do sleep 60; done

echo "[$(agora)] Etapa 3 v2 concluída; medindo a régua do Módulo 3"
./scripts/run_regua_modulo3.sh >> "$LOGS/noite.log" 2>&1
echo "[$(agora)] noite concluída"
