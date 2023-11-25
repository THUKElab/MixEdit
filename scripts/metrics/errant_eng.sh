#!/bin/bash

while getopts "s:h:m:d:r:l:" optname; do
    case $optname in
    s)
        FILE_SRC=${OPTARG};;
    h)
        FILE_HYP=${OPTARG};;
    m)
        FILE_ERRANT=${OPTARG};;
    d)
        DIR_HYP=${OPTARG};;
    r)
        FILE_REF=${OPTARG};;
    l)
        FILE_LOG=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SUFFIX_M2="errant"

FILE_SRC=${FILE_SRC:-"${DIR_DATASET}/conll14st-test.tok.src"}
FILE_REF=${FILE_REF:-"${DIR_DATASET}/conll14st-test.errant"}
FILE_LOG=${FILE_LOG:-"temp.txt"}


if [ -z ${DIR_HYP} ]; then
    if [ -z ${FILE_HYP} ]; then
        echo "No DIR_HYP or FILE_HYP specified. Exit!"
    else
        FILE_ERRANT=${FILE_ERRANT:-${FILE_HYP%.*}.${SUFFIX_M2}}

        echo "#################### ERRANT_ENG Evaluation ####################" | tee -a ${FILE_LOG}
        echo "Source: ${FILE_HYP}" | tee -a ${FILE_LOG}
        echo "Reference: ${FILE_REF}" | tee -a ${FILE_LOG}
        echo "Hypothesis: ${FILE_HYP} -> ${FILE_ERRANT}" | tee -a ${FILE_LOG}

        errant_parallel -orig ${FILE_SRC} -cor ${FILE_HYP} -out ${FILE_ERRANT} | tee -a ${FILE_LOG}
        errant_compare -hyp ${FILE_ERRANT} -ref ${FILE_REF} | tee -a ${FILE_LOG}
    fi
else
    for FILE_HYP in ${DIR_HYP}/pred_*.tgt; do
        FILE_ERRANT=${FILE_HYP%.*}.${SUFFIX_M2}

        echo "#################### ERRANT_ENG Evaluation ####################" | tee -a ${FILE_LOG}
        echo "Source: ${FILE_HYP}" | tee -a ${FILE_LOG}
        echo "Reference: ${FILE_REF}" | tee -a ${FILE_LOG}
        echo "Hypothesis: ${FILE_HYP} -> ${FILE_ERRANT}" | tee -a ${FILE_LOG}

        errant_parallel -orig ${FILE_SRC} -cor ${FILE_HYP} -out ${FILE_ERRANT} | tee -a ${FILE_LOG}
        errant_compare -hyp ${FILE_ERRANT} -ref ${FILE_REF} | tee -a ${FILE_LOG}
    done
fi

