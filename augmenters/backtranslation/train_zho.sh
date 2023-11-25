#!/bin/bash

while getopts "g:i:o:v:m:r:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    v)
        DIR_VALID=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

DIR_INPUT=${DIR_INPUT:-"models/bart/preprocess/zho/hsk+lang8"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_INPUT}"}

DIR_VALID=${DIR_VALID:-"models/bart/preprocess/zho/mucgec_dev"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/zho/hsk+lang8-bart_bt"}

ARCH=${ARCH:-"gec_bart_large"}

MAX_TOKENS=4096

UPDATE_FREQ=4

mkdir -p ${DIR_MODEL} && mkdir -p ${DIR_MODEL}/results

# ========================= Train noisy model =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_OUTPUT}/bin \
    --save-dir ${DIR_MODEL} \
    --user-dir models/bart \
    --restore-file transformers:fnlp/bart-large-chinese \
    --task gec \
    --arch ${ARCH} \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-tokens ${MAX_TOKENS} \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq ${UPDATE_FREQ} \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s tgt \
    -t src \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 30 \
    --patience 5 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --seed 42 >${DIR_MODEL}/nohup.log 2>&1 &

tail -f ${DIR_MODEL}/nohup.log
