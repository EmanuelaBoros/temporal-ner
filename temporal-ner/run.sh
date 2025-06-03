#!/bin/bash

# Check if model_type argument is provided
if [ -z "$1" ]; then
    echo "Please provide a model_type argument: multitask or extended_multitask"
    exit 1
fi

model_type=$1

models=(
    "dbmdz/bert-base-historic-multilingual-cased"
    "bert-base-multilingual-cased"
)

# Array of max sequence lengths to iterate over
max_lengths=(512)

# Temporal fusion strategies
fusion_strategies=("early-cross-attention" "late-cross-attention" "baseline" "concat" "film" "adapter" "relative") # multiscale

# Loop over each model
for model in "${models[@]}"
do
    # Loop over each max sequence length
    for max_len in "${max_lengths[@]}"
    do
        # Loop over each fusion strategy
        for strategy in "${fusion_strategies[@]}"
        do
            echo "Running: model=$model, max_len=$max_len, strategy=$strategy"

            python main.py \
                --model_name_or_path "$model" \
                --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
                --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
                --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
                --output_dir experiments \
                --device cuda \
                --train_batch_size 16 \
                --logging_steps 359 \
                --save_steps 359 \
                --max_sequence_len "$max_len" \
                --evaluate_during_training \
                --do_train \
                --epochs 3 \
                --seed 2024 \
                --model_type "$model_type" \
                --temporal_fusion_strategy "$strategy" --use_relative_year
        done
    done
done