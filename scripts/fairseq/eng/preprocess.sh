#!/bin/bash

while getopts "b:t:i:o:v:" optname; do
    case $optname in
    b)
        DIR_BPE=${OPTARG};;
    t)
        FILE_TSV=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    v)
        DIR_VALID=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

DIR_BPE=${DIR_BPE:-"models/bart/preprocess/eng"}

DIR_VALID=${DIR_VALID:-"${DIR_BPE}/bea_dev"}

DIR_OUTPUT=${DIR_OUTPUT:-${DIR_INPUT}}

mkdir -p ${DIR_OUTPUT}

# Optional: Split TSV file to SRC and TGT files
if [ -f "${FILE_TSV}" ] && [ ! -f "${DIR_INPUT}/src.txt" ]; then
    awk -F'\t' '{print $1}' ${FILE_TSV} > "${DIR_INPUT}/src.txt"
    awk -F'\t' '{print $2}' ${FILE_TSV} > "${DIR_INPUT}/tgt.txt"
fi

# BPE for training dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_OUTPUT}/train.bpe.${LANG}" ]; then
        echo "Apply BPE: ${DIR_INPUT}/${LANG}.txt -> ${DIR_OUTPUT}/train.bpe.${LANG}"
        python ${DIR_BPE}/multiprocessing_bpe_encoder.py \
            --encoder-json "${DIR_BPE}/encoder.json" \
            --vocab-bpe "${DIR_BPE}/vocab.bpe" \
            --inputs "${DIR_INPUT}/${LANG}.txt" \
            --outputs "${DIR_OUTPUT}/train.bpe.${LANG}" \
            --workers 64 \
            --keep-empty
    fi
done

# BPE for validation dataset
for LANG in src tgt; do
    if [ ! -f "${DIR_VALID}/valid.bpe.${LANG}" ]; then
        echo "Apply BPE: ${DIR_VALID}/${LANG}.txt -> ${DIR_VALID}/valid.bpe.${LANG}"
        python ${DIR_BPE}/multiprocessing_bpe_encoder.py \
            --encoder-json "${DIR_BPE}/encoder.json" \
            --vocab-bpe "${DIR_BPE}/vocab.bpe" \
            --inputs "${DIR_VALID}/${LANG}.txt" \
            --outputs "${DIR_VALID}/valid.bpe.${LANG}" \
            --workers 64 \
            --keep-empty
    fi
done

if [ ! -d "${DIR_OUTPUT}/bin" ]; then
    fairseq-preprocess --source-lang "src" --target-lang "tgt" \
        --trainpref "${DIR_OUTPUT}/train.bpe" \
        --validpref "${DIR_VALID}/valid.bpe" \
        --destdir "${DIR_OUTPUT}/bin/" \
        --workers 64 \
        --srcdict "${DIR_BPE}/dict.txt" \
        --tgtdict "${DIR_BPE}/dict.txt"
fi
