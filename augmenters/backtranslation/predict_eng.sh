#!/bin/bash

while getopts "g:p:b:m:n:i:o:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    b)
        DIR_BPE=${DIR_BPE};;
    m)
        DIR_MODEL=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    i)
        FILE_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=${SEED:-"42"}

DIR_BPE=${DIR_BPE:-"models/bart/preprocess/eng"}

DIR_PROCESSED=${DIR_PROCESSED:-"${DIR_BPE}/real/clang8"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/eng/clang8-bart_bt"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best.pt"}

FILE_INPUT=${FILE_INPUT:-""}

DIR_OUTPUT=${DIR_OUTPUT:-""}

FILE_OUTPUT=${DIR_OUTPUT}/$(basename $FILE_INPUT).bt

FILE_LOG=${DIR_OUTPUT}/predict_bt.log

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model : ${FILE_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Input : ${FILE_INPUT}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}
echo "Seed  : ${SEED}" | tee -a ${FILE_LOG}

# Generate Hypothesis:
N_BEST=1
CUDA_VISIBLE_DEVICES=${GPU_list} fairseq-interactive ${DIR_PROCESSED}/bin \
    --task backtranslation -s src -t tgt \
    --user-dir models/backtranslation \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 5 \
    --nbest ${N_BEST} \
    --bpe gpt2 \
    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
    --buffer-size 50000 \
    --batch-size 256 \
    --num-workers 8 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    < ${FILE_INPUT} > ${FILE_OUTPUT}.nbest \
    | tee -a ${FILE_LOG}

cat ${FILE_OUTPUT}.nbest | grep "^D-"  \
    | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % 1 == 0) ]); print(x)" \
    | cut -f 3 > ${FILE_OUTPUT}
sed -i '$d' ${FILE_OUTPUT}



