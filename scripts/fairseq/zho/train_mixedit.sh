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
    r)
        RATE_MASKING=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

# You can also increase or decrease `MAX_TOKENS` according to your GPU memory
MAX_TOKENS=1536

UPDATE_FREQ=12

MIN_FREQ=3

MAX_DIFF=4

DIR_INPUT=${DIR_INPUT:-"models/bart/preprocess/zho/hsk+lang8"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_INPUT}"}

DIR_VALID=${DIR_VALID:-"models/bart/preprocess/zho/mucgec_dev"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/zho/bart_mixedit"}

ARCH=${ARCH:-"gec_bart_large"}

RATE_MASKING=${RATE_MASKING:-"0.05"}

mkdir -p ${DIR_MODEL} && mkdir -p ${DIR_MODEL}/results

# ========================= MixEdit =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_OUTPUT}/bin \
    --save-dir ${DIR_MODEL} \
    --user-dir models/bart \
    --task gec \
    --arch ${ARCH} \
    --restore-file transformers:fnlp/bart-large-chinese \
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
    --warmup-updates 1000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.2 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 10 \
    --patience 5 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
    --augmentation-schema "trg_cut_off" \
    --augmentation-masking-probability ${RATE_MASKING} \
    --augmentation-masking-schema "word" \
    --augmentation-enable-mixedit \
    --mixedit-remove-bpe " ##" \
    --mixedit-temperature 1.0 \
    --augmentation-pattern-noise-rate 0.0 \
    --augmentation-pattern-noise-step 0 \
    --mixedit-filter-pattern-min-freq ${MIN_FREQ} \
    --mixedit-filter-pattern-max-diff ${MAX_DIFF} \
    --file-dataset-m2 "${DIR_INPUT}/mucgec_train.char.m2" \
    --file-pattern "${DIR_INPUT}/pattern.json" \
    --regularization-weight 1.0 \
    --num-workers 8 \
    --seed 42 >${DIR_MODEL}/nohup.log 2>&1 &
wait

# Inference and evaluate
bash scripts/fairseq/zho/predict.sh -g ${GPU_list}\
    -m ${DIR_MODEL} \
    -n "checkpoint_best.pt" \
    -v "mucgec_test"


