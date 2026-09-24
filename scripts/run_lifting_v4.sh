#!/usr/bin/env bash
# Lifting v4: a colocação da junta cortada passa a ser a medida, e não a
# suposta. O v3 prendia tudo na linha de corte; o estimador real só faz isso
# com a junta imediatamente abaixo dela, e espalha as profundas sobre o corpo
# visível (results/posicao_ausentes_*.json).
#
# Ao final mede as duas coisas que decidem: o protocolo de corte sobre o S7,
# que tem ground truth das juntas escondidas, e o Drive&Act, onde a corrupção
# vem do estimador real.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate
LOGS=work_dirs/logs; mkdir -p "$LOGS"

# Registra que esta fila está em execução, para que
# scripts/retomar_apos_reboot.sh saiba o que retomar se a máquina cair. Aquele
# script lê o nome daqui e executa scripts/<nome>, então não precisa conhecer
# esta fila de antemão.
echo "run_lifting_v4.sh" > "$LOGS/fila_pendente"
# O marcador só some quando a fila **conclui**, e é por isso que não há `trap
# EXIT` aqui. O trap dispara também quando o shell é morto no desligamento da
# máquina — exatamente o momento em que o marcador precisa sobreviver. Foi o que
# aconteceu em 12/09: a máquina reiniciou, o trap apagou o marcador antes de
# morrer, e a retomada automática não achou o que religar.
agora() { date '+%H:%M:%S'; }

# Checkpoint de partida do v4, o mesmo que `load_from` do config aponta. Ele é
# também o ponto de comparação, e por isso o nome vive aqui uma vez só.
V2=work_dirs/lift3d_robusto_v2/best_MPJPE_whole_epoch_15.pth

# Mede um checkpoint do lifting no domínio veicular. O recorte, a escala de
# confiança e a normalização precisam ser idênticos entre as chamadas: são eles
# que tornam os números comparáveis entre si, e não o fato de saírem do mesmo
# script.
validar() {
    python scripts/validate_lifting_driveact.py \
        --poses ~/Downloads/extracted/openpose_3d --lift-ckpt "$1" \
        --confidence normalizada \
        --max-sequences 4 --max-frames-per-sequence 120 \
        --out "$2" >> "$LOGS/lifting_v4.log" 2>&1
}

# Espera a GPU esvaziar perguntando quais processos de computação existem, e
# não procurando o processo pelo nome.
#
# `pgrep -f <padrão>` casa com qualquer linha de comando que contenha o padrão,
# **inclusive a do shell que escreveu este arquivo**: o heredoc que o criou tem
# o padrão dentro dele. A fila ficou 2h10 esperando o próprio criador terminar,
# que é algo que não acontece enquanto ela roda.
#
# A lista de processos de computação também é preferível ao total de memória
# usada: o ambiente gráfico segura centenas de MB na mesma GPU o tempo todo, o
# que obriga a inventar um limiar. Processo de computação ou existe ou não.
ocupada() {
    [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
}
while ocupada; do sleep 60; done
echo "[$(agora)] GPU livre"

# O v2 é remedido aqui, antes do treino, por dois motivos que se somam.
#
# O número publicado do v2 (95,77mm de PA-MPJPE, em
# `results/lifting_driveact_v2_corrigido.json`) saiu quando o 2D ainda entrava
# normalizado pela largura do quadro; `--normalizacao` passou a existir depois,
# e hoje o padrão reprojeta a entrada na geometria em que o H3WB treinou.
# Comparar o v4 sob a geometria de treino com o v2 sob a largura mediria a
# correção de escala, não o que o corte ensinou.
#
# E, rodando antes, esta chamada é também o ensaio da validação: se algum
# caminho ou flag do `validate_lifting_driveact.py` mudou, a fila para em três
# minutos em vez de descobrir isso depois de duas horas de treino.
echo "[$(agora)] medindo a linha de base do v2 sob a mesma normalização"
validar "$V2" results/lifting_driveact_v2_camera.json \
    || { echo "[$(agora)] a validação falhou antes do treino; nada foi treinado"; exit 1; }

echo "[$(agora)] treinando o lifting v4"

# `--resume` aqui não força retomada: `scripts/train_wholebody.py` decide pelo
# que existe no work_dir, e cai para o `load_from` do config quando não há nada
# a retomar. É o que torna a fila segura de religar depois de um travamento.
for tentativa in 1 2 3; do
    python scripts/train_wholebody.py \
        --config configs/lift3d_dstformer_h3wb_robusto_v4.py --resume \
        >> "$LOGS/lifting_v4.log" 2>&1 && break
    echo "[$(agora)] treino falhou (tentativa $tentativa)"; sleep 30
done

MELHOR=$(ls -t work_dirs/lift3d_robusto_v4/best_MPJPE_whole_*.pth 2>/dev/null | head -1)
[ -z "$MELHOR" ] && { echo "[$(agora)] sem checkpoint"; exit 1; }
echo "[$(agora)] melhor checkpoint: $MELHOR"

echo "[$(agora)] validando no domínio veicular"
validar "$MELHOR" results/lifting_driveact_v4.json

# A última época é medida junto porque o `best` é escolhido pelo MPJPE na
# validação S7 **sem corte**, herdado do config base. Com metade das janelas de
# treino cortadas, o melhor no limpo não é necessariamente o melhor sob corte —
# pode ser uma época inicial, ainda quase idêntica ao v2. Medir as duas custa
# três minutos e impede que a conclusão dependa de um seletor que mede outra
# coisa.
ULTIMA=$(ls -t work_dirs/lift3d_robusto_v4/epoch_*.pth 2>/dev/null | head -1)
if [ -n "$ULTIMA" ] && [ "$ULTIMA" != "$MELHOR" ]; then
    echo "[$(agora)] validando também a última época: $ULTIMA"
    validar "$ULTIMA" results/lifting_driveact_v4_ultima.json
fi
echo "[$(agora)] protocolo de corte sobre o S7"
python scripts/measure_lifting_truncation.py \
    --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v4.py \
    --lift-ckpt "$MELHOR" --tag v4 --unobserved-confidence 0.3 \
    >> "$LOGS/lifting_v4.log" 2>&1

echo "[$(agora)] qualidade ao vivo na gravação com tabuleiro"
python scripts/measure_live_quality.py --tag v4_ao_vivo --distancia 1.11 \
    --video work_dirs/panel/rec_20260921_224039.mp4 \
    --lift-cfg configs/lift3d_dstformer_h3wb_robusto_v4.py \
    --lift-ckpt "$MELHOR" >> "$LOGS/lifting_v4.log" 2>&1

echo "[$(agora)] concluído"

rm -f "$LOGS/fila_pendente"
