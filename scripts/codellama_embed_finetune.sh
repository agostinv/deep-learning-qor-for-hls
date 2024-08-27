#!/bin/bash

export ROOT=<PATH_TO_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export DATA_DIR=$ROOT/data

cd ${ROOT}
source ${VENV_ROOT}/bin/activate

export PYTHONPATH=${PYTHONPATH}:"./"

python ${ROOT}/src/trainers/codellama_embed_finetune \
    --data-path ${DATA_DIR} \
    --model-path "codellama/CodeLlama-7b-hf" \
    --lora-r 8 --lora-apha 32 --lora-dropout 0.1 \
    --compute-dtype float32 \
    --output-path "codellama-7b-embedding-finetune-test" \