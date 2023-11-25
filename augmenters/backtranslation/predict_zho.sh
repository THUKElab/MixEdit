#!/bin/bash

while getopts "g:p:b:c:m:o:n:i:s:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    i)
        FILE_INPUT=${OPTARG};;
    s)
        SEED=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=${SEED:-"42"}

DIR_PROCESSED=${DIR_PROCESSED:-"models/bart/preprocess/zho/hsk+lang8"}

DIR_MODEL=${DIR_MODEL:-"models/bart/exps/zho/hsk+lang8-bart_bt"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best.pt"}

FILE_INPUT=${FILE_INPUT:-""}

DIR_OUTPUT=${DIR_OUTPUT:-""}

FILE_OUTPUT=${DIR_OUTPUT}/$(basename $FILE_INPUT).bt

FILE_LOG=${DIR_OUTPUT}/predict_bt.log

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model : ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Input : ${FILE_INPUT}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}
echo "Seed  : ${SEED}" | tee -a ${FILE_LOG}

# Generate Hypothesis
CUDA_VISIBLE_DEVICES=${GPU_list} fairseq-interactive ${DIR_PROCESSED}/bin \
    --task backtranslation -s src -t tgt \
    --user-dir models/backtranslation \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 5 \
    --nbest 1 \
    --buffer-size 10000 \
    --batch-size 128 \
    --num-workers 4 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --seed ${SEED} \
    < ${FILE_INPUT} > ${FILE_OUTPUT}.nbest | tee -a ${FILE_LOG}

cat ${FILE_OUTPUT}.out.nbest | grep "^D-"  \
    | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % 1 == 0) ]); print(x)" \
    | cut -f 3 > ${FILE_OUTPUT}
sed -i '$d' ${FILE_OUTPUT}


