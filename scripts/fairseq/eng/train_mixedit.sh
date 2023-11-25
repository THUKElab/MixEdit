#!/bin/bash

while getopts "g:i:b:m:a:r:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    b)
        DIR_BPE=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    a)
        ARCH=${OPTARG};;
    r)
        MASK_RATE=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

echo "GPU: ${GPU_list}"

SEED=42

# You can also increase or decrease `MAX_TOKENS` according to your GPU memory
MAX_TOKENS=4096

UPDATE_FREQ=4

MIN_FREQ=-1

MAX_DIFF=-1

REG_WEIGHT=1.0

ARCH=${ARCH:-"gec_bart_large"}

BART_PATH="../resources/bart.large/model.pt"

MASK_RATE=${MASK_RATE:-"0.20"}

MIXEDIT_TEMPERATURE=${MIXEDIT_TEMPERATURE:-"1.0"}

DIR_MODEL=${DIR_MODEL:-"bart_mixedit"}

DIR_BPE=${DIR_BPE:-"models/bart/preprocess/eng"}

DIR_MODEL_STAGE1=models/bart/exps/eng/mixedit/stage1/${DIR_MODEL}
DIR_MODEL_STAGE2=models/bart/exps/eng/mixedit/stage2/${DIR_MODEL}
DIR_MODEL_STAGE3=models/bart/exps/eng/mixedit/stage3/${DIR_MODEL}

DIR_INPUT_STAGE1=models/bart/preprocess/eng/real/clang8
DIR_INPUT_STAGE2=models/bart/preprocess/eng/real/bea_train1
DIR_INPUT_STAGE3=models/bart/preprocess/eng/real/bea_train2

mkdir -p ${DIR_MODEL_STAGE1}
mkdir -p ${DIR_MODEL_STAGE2}
mkdir -p ${DIR_MODEL_STAGE3}
mkdir -p ${DIR_MODEL_STAGE1}/results
mkdir -p ${DIR_MODEL_STAGE2}/results
mkdir -p ${DIR_MODEL_STAGE3}/results


# ========================= Stage 1 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_INPUT_STAGE1}/bin \
    --save-dir ${DIR_MODEL_STAGE1} \
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
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
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
    --mixedit-filter-pattern-min-freq ${MIN_FREQ} \
    --mixedit-filter-pattern-max-diff ${MAX_DIFF} \
    --mixedit-regularization-weight ${REG_WEIGHT} \
    --file-dataset-m2 "${DIR_BPE}/clang8/clang8.errant" \
    --file-pattern "${DIR_BPE}/merge_pattern.json" \
    --num-workers 8 \
    --seed $SEED >${DIR_MODEL_STAGE1}/nohup.log 2>&1 &
wait

# ========================= Stage 2 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_INPUT_STAGE2}/bin \
    --save-dir ${DIR_MODEL_STAGE2} \
    --user-dir models/bart \
    --arch ${ARCH} \
    --task gec \
    --finetune-from-model ${DIR_MODEL_STAGE1}/checkpoint_best.pt \
    --max-tokens ${MAX_TOKENS} \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --weight-decay 0.01 \
    --update-freq ${UPDATE_FREQ} \
    --lr 5e-06 \
    --warmup-updates 100 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --patience 10 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
    --bpe gpt2 \
    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
    --augmentation-schema "trg_cut_off" \
    --augmentation-masking-probability ${MASK_RATE} \
    --augmentation-masking-schema "word" \
    --augmentation-enable-mixedit \
    --mixedit-temperature ${MIXEDIT_TEMPERATURE} \
    --mixedit-filter-pattern-min-freq ${MIN_FREQ} \
    --mixedit-filter-pattern-max-diff ${MAX_DIFF} \
    --mixedit-regularization-weight ${REG_WEIGHT} \
    --file-dataset-m2 "${DIR_BPE}/bea_train1/bea_train1.errant" \
    --file-pattern "${DIR_BPE}/merge_pattern.json" \
    --num-workers 8 \
    --seed $SEED >${DIR_MODEL_STAGE2}/nohup.log 2>&1 &
wait

# ========================= Stage 3 =========================
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/train.py ${DIR_INPUT_STAGE3}/bin \
    --save-dir ${DIR_MODEL_STAGE3} \
    --user-dir models/bart \
    --arch ${ARCH} \
    --task gec \
    --finetune-from-model ${DIR_MODEL_STAGE2}/checkpoint_best.pt \
    --max-tokens ${MAX_TOKENS} \
    --optimizer adam \
    --layernorm-embedding \
    --weight-decay 0.01 \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq ${UPDATE_FREQ} \
    --lr 3e-06 \
    --warmup-updates 50 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 20 \
    --patience 5 \
    --adam-betas "(0.9,0.999)" \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 30 \
    --bpe gpt2 \
    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
    --augmentation-schema "trg_cut_off" \
    --augmentation-masking-probability ${MASK_RATE} \
    --augmentation-masking-schema "word" \
    --augmentation-enable-mixedit \
    --mixedit-temperature ${MIXEDIT_TEMPERATURE} \
    --mixedit-filter-pattern-min-freq ${MIN_FREQ} \
    --mixedit-filter-pattern-max-diff ${MAX_DIFF} \
    --mixedit-regularization-weight ${REG_WEIGHT} \
    --file-dataset-m2 "${DIR_BPE}/bea_train2/bea_train2.errant" \
    --file-pattern "${DIR_BPE}/merge_pattern.json" \
    --num-workers 8 \
    --seed $SEED >${DIR_MODEL_STAGE3}/nohup.log 2>&1 &
wait

# Inference and evaluate
bash scripts/fairseq/eng/predict.sh -g ${GPU_list} \
    -m ${DIR_MODEL_STAGE3} \
    -n "checkpoint_best.pt" \
    -v "bea_dev"

