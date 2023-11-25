#!/bin/bash

while getopts "g:p:b:m:n:i:o:r:v:l:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    b)
        DIR_BPE=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    i)
        FILE_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    r)
        FILE_REF=${OPTARG};;
    v)
        VALID_NAME=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

if [ -z ${DIR_MODEL} ]; then
    echo "DIR_MODEL not exists"
    exit -1
fi

DIR_PROCESSED=${DIR_PROCESSED:-"models/bart/preprocess/eng/clang8/bin"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best.pt"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_MODEL}/results"}

FILE_LOG="${DIR_OUTPUT}/${VALID_NAME}.log"

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

# Default input and reference files for validation
if [ ${VALID_NAME} = "bea_dev" ]; then
    FILE_INPUT=${FILE_INPUT:-"models/bart/preprocess/eng/bea_dev/src.txt"}
    FILE_REF=${FILE_REF:-"models/bart/preprocess/eng/bea_dev/bea_dev.errant"}
fi

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model: ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Valid: ${VALID_NAME}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}


# Generate Hypothesis
N_BEST=1
CUDA_VISIBLE_DEVICES=${GPU_list} fairseq-interactive ${DIR_PROCESSED} \
    --task translation \
    --user-dir models/bart \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 12 \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --bpe gpt2 \
    --buffer-size 10000 \
    --batch-size 128 \
    --num-workers 8 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    < ${FILE_INPUT} > ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | tee -a ${FILE_LOG}

cat ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | grep "^D-"  \
    | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" \
    | cut -f 3 > ${DIR_OUTPUT}/${VALID_NAME}.out
sed -i '$d' ${DIR_OUTPUT}/${VALID_NAME}.out


# Post-process
if [ ${VALID_NAME} = "bea_dev" ]; then
    # ERRANT Evaluation
    echo "Errant Evaluation" | tee -a ${FILE_LOG}
    bash scripts/metrics/errant_eng.sh -s ${FILE_INPUT} -h ${DIR_OUTPUT}/${VALID_NAME}.out -r ${FILE_REF} -l ${FILE_LOG}
elif [ ${VALID_NAME} = "conll14" ]; then
    echo "Post-process CoNLL-2014: 80" | tee -a ${FILE_LOG}
    python processors/post_process_eng.py ${FILE_INPUT} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_process   80
elif [ ${VALID_NAME} = "bea_test" ]; then
    echo "Post-process BEA-2019 Test: 64" | tee -a ${FILE_LOG}
    python processors/post_process_eng.py ${FILE_INPUT} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_process   64
fi
