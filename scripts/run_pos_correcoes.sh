#!/usr/bin/env bash
# Depois das correções da noite de 23/09: a bateria de validação com o lifting em
# float16 e o modelo veicular corrigido, e a figura do painel refeita.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=work_dirs/logs/pos_correcoes.log
agora() { date '+%H:%M:%S'; }
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
ocupada() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while ocupada; do sleep 30; done
echo "[$(agora)] GPU livre"

echo "[$(agora)] bateria de validação, lifting em float16"
python scripts/run_validation_battery.py --out results/bateria_validacao.json >> "$LOG" 2>&1

echo "[$(agora)] figura do painel, montagem de retrovisor"
V=~/Downloads/extracted/inner_mirror/vp14/run1_2018-05-30-10-11-09.ids_1
python scripts/render_panel_frame.py --quadro 400 \
    --saida ../../Projeto-Fisico/projeto-fisico/Image/painel_retrovisor.png -- \
    --montagem retrovisor --source "$V.mp4" --calibracao "$V.calibration.json" >> "$LOG" 2>&1
echo "[$(agora)] concluído"
