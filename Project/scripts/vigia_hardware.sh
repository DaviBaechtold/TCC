#!/usr/bin/env bash
# Vigia a fila de treino e o hardware, e recupera o que dá para recuperar.
#
# Esta máquina cai de três maneiras distintas, e cada uma exige uma resposta:
#
#   1. Travamento total do kernel. Nenhum software interno responde. Quem
#      resolve é o watchdog da placa (`iTCO_wdt` mais `RuntimeWatchdogSec=60s`
#      no systemd), que reinicia por hardware; o `@reboot` do crontab retoma a
#      fila. Esta vigia não tem papel aqui.
#   2. A GPU cai do barramento com o sistema de pé --- `nvidia-smi` devolve
#      "Unable to determine the device handle". Aconteceu em 21/09/2026. O
#      kernel está vivo, então o watchdog não dispara, e só um reinício
#      recupera. É o que esta vigia tenta.
#   3. A fila morre sozinha: erro de CUDA transitório, OOM, processo morto. A
#      GPU continua sadia e basta relançar --- o treino retoma da última época
#      pelo `--resume`, e a fila registra o progresso em arquivo.
#
# Dois modos. Sem argumento faz uma verificação e sai, que é a forma de rodar
# por cron. Com um intervalo em segundos, fica em laço até a fila concluir ---
# é assim que ela sobe junto da fila e pelo `retomar_apos_reboot.sh`, sem
# depender de crontab.
#
#   ./scripts/vigia_hardware.sh          uma verificação
#   ./scripts/vigia_hardware.sh 300      verifica a cada cinco minutos
set -u
cd "$(dirname "$0")/.."

INTERVALO=${1:-0}
if [ "$INTERVALO" -gt 0 ]; then
    # O laço morre junto com a fila: sem marcador não há o que vigiar, e uma
    # vigia órfã agiria por causa de um treino que já acabou.
    while [ -f work_dirs/logs/fila_pendente ]; do
        "$0"
        sleep "$INTERVALO"
    done
    exit 0
fi

LOGS=work_dirs/logs
mkdir -p "$LOGS"
PENDENTE="$LOGS/fila_pendente"
DIARIO="$LOGS/vigia.log"
FALHAS="$LOGS/vigia_falhas_gpu"
REINICIOS="$LOGS/vigia_reinicios"

# Sem fila pendente não há o que proteger: a vigia não reinicia uma máquina
# ociosa só porque a GPU está estranha.
[ -f "$PENDENTE" ] || exit 0
FILA=$(cat "$PENDENTE")

# Duas leituras ruins seguidas, e não uma: `nvidia-smi` falha esporadicamente
# sob carga pesada sem que a GPU tenha caído.
FALHAS_PARA_REINICIAR=2
# Teto de reinícios, para que um defeito permanente não vire laço de boot.
MAXIMO_DE_REINICIOS=3

anotar() { echo "[$(date '+%F %H:%M:%S')] $*" >> "$DIARIO"; }

if ! nvidia-smi -L > /dev/null 2>&1; then
    falhas=$(( $(cat "$FALHAS" 2>/dev/null || echo 0) + 1 ))
    echo "$falhas" > "$FALHAS"
    anotar "GPU não responde ($falhas leitura(s) seguida(s))"

    [ "$falhas" -lt "$FALHAS_PARA_REINICIAR" ] && exit 0

    reinicios=$(cat "$REINICIOS" 2>/dev/null || echo 0)
    if [ "$reinicios" -ge "$MAXIMO_DE_REINICIOS" ]; then
        anotar "teto de $MAXIMO_DE_REINICIOS reinícios atingido; parando de tentar"
        exit 0
    fi
    echo $(( reinicios + 1 )) > "$REINICIOS"
    anotar "reiniciando a máquina para recuperar a GPU; a fila $FILA retoma pelo @reboot"
    sync
    # Duas tentativas: a do systemd, que depende de autorização do polkit, e a
    # via sudo, que depende da regra sem senha em /etc/sudoers.d. Se nenhuma
    # passar, fica registrado para a manhã seguinte.
    systemctl reboot 2>>"$DIARIO" || sudo -n /usr/sbin/reboot 2>>"$DIARIO" \
        || anotar "SEM PERMISSÃO PARA REINICIAR: reinicie à mão para retomar $FILA"
    exit 0
fi

rm -f "$FALHAS"

# GPU sadia. Se a fila sumiu mas o marcador continua lá, ela morreu sem
# concluir: relançar é seguro porque o treino retoma da última época.
if ! pgrep -f "scripts/$FILA" > /dev/null 2>&1; then
    anotar "fila $FILA ausente com marcador presente; relançando"
    setsid "./scripts/$FILA" >> "$LOGS/${FILA%.sh}_vigia.log" 2>&1 < /dev/null &
fi
