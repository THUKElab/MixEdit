#!/bin/bash

while getopts "g:i:o:b:v:m:l:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    b)
        DIR_BPE=${OPTARG};;
    v)
        DIR_VALID=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    l)
        FILE_LOG=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

echo "GPU: ${GPU_list}"

SEED=42

BART_PATH="../resources/bart.large/model.pt"

DIR_BPE=${DIR_BPE:-"models/bart/preprocess/eng"}

DIR_INPUT=${DIR_INPUT:-"${DIR_BPE}/clang8"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_INPUT}"}

DIR_VALID=${DIR_VALID:-"${DIR_BPE}/bea_dev"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/eng/clang8-bart_bt"}

mkdir -p ${DIR_MODEL} && mkdir -p ${DIR_MODEL}/results

# ========================= Train noisy model =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup fairseq-train ${DIR_OUTPUT}/bin \
    --save-dir ${DIR_MODEL} \
    --arch bart_large \
    --restore-file ${BART_PATH} \
    --task translation \
    --max-tokens 4096 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --weight-decay 0.01 \
    --update-freq 4 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    -s tgt \
    -t src \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 100 \
    --patience 10 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --seed $SEED >${DIR_MODEL}/nohup.log 2>&1 &

tail -f ${DIR_MODEL}/nohup.log
