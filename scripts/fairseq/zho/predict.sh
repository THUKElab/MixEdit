#!/bin/bash

while getopts "g:p:m:n:i:o:r:v:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    i)
        FILE_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
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

DIR_PROCESSED=${DIR_PROCESSED:-"models/bart/preprocess/zho/real/hsk+lang8/bin"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_MODEL}/results"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best.pt"}

FILE_LOG="${DIR_OUTPUT}/${VALID_NAME}.log"

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

# Default input and inference file for validation
if [ ${VALID_NAME} = "mucgec_dev" ]; then
    FILE_ID="models/bart/preprocess/zho/mucgec_dev/mucgec_dev.seg.id"
    FILE_INPUT=${FILE_INPUT:-"models/bart/preprocess/zho/mucgec_dev/mucgec_dev.seg.char.src"}
elif [ ${VALID_NAME} = "mucgec_test" ]; then
    FILE_ID="models/bart/preprocess/zho/mucgec_test/mucgec_test.seg.id"
    FILE_INPUT=${FILE_INPUT:-"models/bart/preprocess/zho/mucgec_test/mucgec_test.seg.char.src"}
elif [ ${VALID_NAME} = "nlpcc_test" ]; then
    FILE_ID="models/bart/preprocess/zho/nlpcc_test/nlpcc_test.seg.id"
    FILE_INPUT=${FILE_INPUT:-"models/bart/preprocess/zho/nlpcc_test/nlpcc_test.seg.char.src"}
fi

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model: ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Valid: ${VALID_NAME}" | tee -a ${FILE_LOG}
echo "Input: ${FILE_INPUT}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}

# Generate Hypothesis
CUDA_VISIBLE_DEVICES=${GPU_list} fairseq-interactive ${DIR_PROCESSED} \
    --task translation \
    --user-dir models/bart \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 10 \
    --nbest 1 \
    --max-len-b 200 \
    -s src \
    -t tgt \
    --buffer-size 10000 \
    --batch-size 64 \
    --num-workers 8 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    < ${FILE_INPUT} > ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | tee -a ${FILE_LOG}

cat ${DIR_OUTPUT}/${VALID_NAME}.out.nbest | grep "^D-"  \
    | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % 1 == 0) ]); print(x)" \
    | cut -f 3 > ${DIR_OUTPUT}/${VALID_NAME}.out
sed -i '$d' ${DIR_OUTPUT}/${VALID_NAME}.out


# Evaluation
if [ ${VALID_NAME} = "mucgec_dev" ]; then
    echo "Post-process mucgec_dev" | tee -a ${FILE_LOG}
    FILE_REF="models/bart/preprocess/zho/mucgec_dev/mucgec_dev.errant"
    FILE_SRC="models/bart/preprocess/zho/mucgec_dev/mucgec_dev.src"
    FILE_SEG_SRC="models/bart/preprocess/zho/mucgec_dev/mucgec_dev.seg.src"

    python processors/post_process_zho.py ${FILE_SEG_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    # ERRANT Evaluation
    echo "ERRANT Evaluation" | tee -a ${FILE_LOG}
    bash scripts/metrics/errant_zho.sh \
        -s ${FILE_SRC} \
        -h ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        -r ${FILE_REF} -l ${FILE_LOG}

elif [ ${VALID_NAME} = "mucgec_test" ]; then
    echo "Post-process mucgec_test" | tee -a ${FILE_LOG}
    FILE_SRC="models/bart/preprocess/zho/mucgec_test/mucgec_test.src"
    FILE_SEG_SRC="models/bart/preprocess/zho/mucgec_test/mucgec_test.seg.src"

    python processors/post_process_zho.py ${FILE_SEG_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 10000

    paste ${FILE_SRC} ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        | awk '{print NR"\t"$p}' > ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed.para

elif [ ${VALID_NAME} = "nlpcc_test" ]; then
    echo "Post-process nlpcc_test" | tee -a ${FILE_LOG}
    FILE_REF="models/bart/preprocess/zho/nlpcc_test/gold.01"
    FILE_SRC="models/bart/preprocess/zho/nlpcc_test/nlpcc_test.src"
    FILE_SEG_SRC="models/bart/preprocess/zho/nlpcc_test/nlpcc_test.seg.src"

    python processors/post_process_zho.py ${FILE_SEG_SRC} \
        ${DIR_OUTPUT}/${VALID_NAME}.out \
        ${DIR_OUTPUT}/${VALID_NAME}.out.post_processed \
        ${FILE_ID} 128

    # MaxMatch Evaluation
    echo "MaxMatch Evaluation (TODO)" | tee -a ${FILE_LOG}
fi

