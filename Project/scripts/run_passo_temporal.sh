#!/usr/bin/env bash
# Janela do lifting ao vivo com passo 1 contra passo 3, na mesma gravação.
#
# O lifting aprendeu contexto no H3WB, com 100ms medianos entre quadros da
# janela e 3,7s de duração; a 30 FPS o painel monta janelas de 33ms e 0,5s.
# Passo 3 reproduz o intervalo do treino sem atrasar a saída.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=work_dirs/logs/passo_temporal.log
agora() { date '+%H:%M:%S'; }

# Espera a fila anterior pelo processo dela, recebido como argumento.
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
ocupada() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while ocupada; do sleep 30; done
echo "[$(agora)] GPU livre"

# Regressão: com passo 1 o caminho ao vivo precisa continuar reproduzindo a
# validação.
python tests/test_sequence_lifter.py >> "$LOG" 2>&1 \
    && echo "[$(agora)] teste do lifting passa" \
    || echo "[$(agora)] TESTE DO LIFTING FALHOU"

for passo in 1 3; do
    echo "[$(agora)] qualidade ao vivo, passo $passo"
    python scripts/measure_live_quality.py --tag "passo${passo}_v3" \
        --passo-temporal "$passo" --distancia 1.11 \
        --video work_dirs/panel/rec_20260921_224039.mp4 >> "$LOG" 2>&1
done
echo "[$(agora)] concluído"
