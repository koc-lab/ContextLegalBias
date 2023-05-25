# run_main.sh:  Bash script to run "main.py"
#               LR/EPS/EPOCH for GAP-Flipped:  [1e-5:1e-4] / 1e-8 / 3
#                            for ECtHR-GTuned: [1e-5:1e-4] / 1e-8 / 3
# Author:       Mustafa Bozdag
# Date:         04/28/2023

TUNEDATA="./data/ecthr-gtuned.tsv"
EVALDATA="./data/bec-cri.tsv"
LR=1e-5
EPS=1e-8
EPOCH=3
GPUID=0
MODELDIR="nlpaueb/legal-bert-small-uncased"

python ./main.py --model ${MODELDIR} --eval ${EVALDATA} --lr ${LR} --eps ${EPS} --epoch ${EPOCH} --devID ${GPUID}