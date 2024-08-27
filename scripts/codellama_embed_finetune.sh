#!/bin/bash

export ROOT=<PATH_TO_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export DATA_DIR=$ROOT/data/train_data

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

cd ${ROOT}
source ${VENV_ROOT}/bin/activate

export PYTHONPATH=${PYTHONPATH}:"./:./src"

python ${ROOT}/src/trainers/codellama_embed_finetune.py \
    --data-path ${DATA_DIR} \
    --model-name "codellama/CodeLlama-7b-hf" \
    --lora-r 8 --lora-alpha 32 --lora-dropout 0.1 \
    --compute-dtype float32 \
    --output-path "checkpoints/codellama-7b-embedding-finetune-test" \