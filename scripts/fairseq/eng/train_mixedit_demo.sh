#!/bin/bash

while getopts "g:i:o:b:v:m:" optname; do
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
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

# You can also increase or decrease `MAX_TOKENS` according to your GPU memory
MAX_TOKENS=4096

UPDATE_FREQ=4

ARCH=${ARCH:-"gec_bart_large"}

BART_PATH="../resources/bart.large/model.pt"

MASK_RATE=${MASK_RATE:-"0.20"}

MIXEDIT_TEMPERATURE=${MIXEDIT_TEMPERATURE:-"1.0"}

DIR_BPE=${DIR_BPE:-"models/bart/preprocess/eng"}

DIR_INPUT=${DIR_INPUT:-"${DIR_BPE}/bea_train2"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_INPUT}"}

DIR_VALID=${DIR_VALID:-"${DIR_BPE}/bea_dev"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/eng/bea_train2-bart_mixedit"}

mkdir -p ${DIR_MODEL} && mkdir -p ${DIR_MODEL}/results

if [ ! -d ${DIR_OUTPUT}/bin ]; then
    bash scripts/fairseq/eng/preprocess.sh -i ${DIR_INPUT}  -o ${DIR_OUTPUT}  -v ${DIR_VALID}
fi

# ========================= MixEdit =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_OUTPUT}/bin \
    --save-dir ${DIR_MODEL} \
    --user-dir models/bart \
    --arch ${ARCH} \
    --task gec \
    --restore-file ${BART_PATH} \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-tokens ${MAX_TOKENS} \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq ${UPDATE_FREQ} \
    --lr 3e-05 \
    --warmup-updates 100 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 10 \
    --patience 10 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --bpe gpt2 \
    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
    --augmentation-schema "trg_cut_off" \
    --augmentation-masking-probability ${MASK_RATE} \
    --augmentation-masking-schema "word" \
    --augmentation-enable-mixedit \
    --mixedit-temperature ${MIXEDIT_TEMPERATURE} \
    --mixedit-regularization-weight 1.0 \
    --file-dataset-m2 "${DIR_BPE}/bea_train2/bea_train2.errant" \
    --file-pattern "${DIR_BPE}/merge_pattern.json" \
    --num-workers 8 \
    --seed 42 >${DIR_MODEL}/nohup.log 2>&1 &
wait

# Inference and evaluate
bash scripts/fairseq/eng/predict.sh -g ${GPU_list} \
    -m ${DIR_MODEL_STAGE3} \
    -n "checkpoint_best.pt" \
    -v "bea_dev"
