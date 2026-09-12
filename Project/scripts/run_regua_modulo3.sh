#!/usr/bin/env bash
# Compara os três lifting pela régua que não depende da referência ruidosa.
#
# A pose 3D do Drive&Act tem ao menos 30,1mm de incerteza por quadro, mais que o
# dobro da diferença entre os modelos. A coerência de comprimento de osso não
# usa referência alguma: o osso é fisicamente constante, então toda variação na
# pose predita é erro do modelo.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair.
echo "run_regua_modulo3.sh" > "$LOGS/fila_pendente"

agora() { date '+%H:%M:%S'; }

ocupada() {
    local usada
    usada=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${usada:-0}" -gt 1500 ]
}
while ocupada; do sleep 60; done
echo "[$(agora)] GPU livre; medindo a coerência dos três modelos"

declare -A MODELOS=(
  [base]=work_dirs/lift3d_dstformer_h3wb/best_MPJPE_whole_epoch_30.pth
  [apagamento]=work_dirs/lift3d_robusto/best_MPJPE_whole_epoch_13.pth
  [simulacao]=work_dirs/lift3d_robusto_v2/best_MPJPE_whole_epoch_15.pth
)
for nome in base apagamento simulacao; do
    ck="${MODELOS[$nome]}"
    [ -f "$ck" ] || { echo "[$(agora)] $nome: checkpoint ausente"; continue; }
    echo "[$(agora)] medindo $nome"
    python scripts/validate_lifting_driveact.py \
        --poses ~/Downloads/extracted/openpose_3d --lift-ckpt "$ck" \
        --confidence normalizada --max-sequences 6 --max-frames-per-sequence 200 \
        --out "results/regua_$nome.json" >> "$LOGS/regua_modulo3.log" 2>&1
done

echo "[$(agora)] concluído"
rm -f "$LOGS/fila_pendente"
