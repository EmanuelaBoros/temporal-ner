#!/bin/bash

# General English models
MULTILINGUAL_MODELS=("bert-base-multilingual-cased" "bert-base-multilingual-uncased" "dbmdz/bert-base-historic-multilingual-cased" "dbmdz/bert-medium-historic-multilingual-cased" "dbmdz/bert-mini-historic-multilingual-cased")

# Mapping of full language names to their shorter forms
declare -A LANGUAGE_MAPPING
LANGUAGE_MAPPING["french"]="fr"
LANGUAGE_MAPPING["english"]="en"
LANGUAGE_MAPPING["german"]="de"
LANGUAGE_MAPPING["greek"]="el"  # Assuming 'el' for Greek; adjust if necessary
LANGUAGE_MAPPING["finnish"]="fi"
LANGUAGE_MAPPING["swedish"]="sv"

# Language-specific models for each dataset and language
declare -A LANGUAGE_SPECIFIC_MODELS

LANGUAGE_SPECIFIC_MODELS["ajmc,german"]="bert-base-german-cased,dbmdz/bert-base-german-europeana-cased,dbmdz/bert-base-german-europeana-uncased"
LANGUAGE_SPECIFIC_MODELS["ajmc,greek"]="pranaydeeps/Ancient-Greek-BERT,nlpaueb/bert-base-greek-uncased-v1"
LANGUAGE_SPECIFIC_MODELS["ajmc,french"]="camembert-base,dbmdz/bert-base-french-europeana-cased"
LANGUAGE_SPECIFIC_MODELS["ajmc,english"]="bert-base-cased,bert-base-uncased"

LANGUAGE_SPECIFIC_MODELS["hipe2020,french"]="camembert-base,dbmdz/bert-base-french-europeana-cased"

LANGUAGE_SPECIFIC_MODELS["letemps,french"]="camembert-base,dbmdz/bert-base-french-europeana-cased"

LANGUAGE_SPECIFIC_MODELS["newseye,french"]="camembert-base,dbmdz/bert-base-french-europeana-cased"
LANGUAGE_SPECIFIC_MODELS["newseye,german"]="bert-base-german-cased,dbmdz/bert-base-german-europeana-cased,dbmdz/bert-base-german-europeana-uncased"
LANGUAGE_SPECIFIC_MODELS["newseye,finnish"]="TurkuNLP/bert-base-finnish-cased-v1,TurkuNLP/bert-base-finnish-uncased-v1"
LANGUAGE_SPECIFIC_MODELS["newseye,swedish"]="KB/bert-base-swedish-cased"

LANGUAGE_SPECIFIC_MODELS["sonar,german"]="bert-base-german-cased,dbmdz/bert-base-german-europeana-cased,dbmdz/bert-base-german-europeana-uncased"

LANGUAGE_SPECIFIC_MODELS["topres19th,german"]="bert-base-cased,bert-base-uncased"

DATASETS=("hipe2020" "ajmc" "letemps" "newseye" "sonar" "topres19th")

# If no GPU is specified, default to 0. If no language is specified, it will run for all languages.
GPU=${1:-0}
SPECIFIC_LANGUAGE=${2:-""}

# Base command
BASE_CMD="TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU python main.py --batch_size 8 --output_dir ./experiments/ --do_train --model bert --dataset_dir ../data/ --device cuda"

for DATASET in "${DATASETS[@]}"
do
    # Determine the languages based on the dataset
    declare -a CURRENT_LANGUAGES
    case "$DATASET" in
        "ajmc")
            CURRENT_LANGUAGES=("german" "greek" "french" "english")
            ;;
        "hipe2020")
            CURRENT_LANGUAGES=("french")
            ;;
        "letemps")
            CURRENT_LANGUAGES=("french")
            ;;
        "newseye")
            CURRENT_LANGUAGES=("french" "german" "finnish" "swedish")
            ;;
        "sonar")
            CURRENT_LANGUAGES=("german")
            ;;
        "topres19th")
            CURRENT_LANGUAGES=("english")
            ;;
        *)
            # For any unanticipated dataset, just continue to the next iteration
            continue
            ;;
    esac

    # If a specific language was provided, check if it's in the current dataset's languages
    if [ ! -z "$SPECIFIC_LANGUAGE" ]; then
        # shellcheck disable=SC2076
        # shellcheck disable=SC2076
        # shellcheck disable=SC2199
        if [[ ! "${CURRENT_LANGUAGES[@]}" =~ "${SPECIFIC_LANGUAGE}" ]]; then
            # The specific language isn't in this dataset's languages, so skip this dataset
            continue
        else
            # Overwrite CURRENT_LANGUAGES with only the specific language
            CURRENT_LANGUAGES=("$SPECIFIC_LANGUAGE")
        fi
    fi

    for LANGUAGE in "${CURRENT_LANGUAGES[@]}"
    do
        # Start with the Multilingual models
        MODELS_FOR_THIS_LANG=("${MULTILINGUAL_MODELS[@]}")

        # Add language-specific models for this dataset and language
        KEY="$DATASET,$LANGUAGE"
        if [[ -n "${LANGUAGE_SPECIFIC_MODELS[$KEY]}" ]]; then
            IFS=',' read -ra SPECIFIC_MODELS_FOR_LANGUAGE <<< "${LANGUAGE_SPECIFIC_MODELS[$KEY]}"
        fi

        # Get the short form of the language for the file paths
        SHORT_LANGUAGE=${LANGUAGE_MAPPING["$LANGUAGE"]}


        # Start with the multilingual models
        MODELS_FOR_THIS_LANG=("${MULTILINGUAL_MODELS[@]}")
        for MODEL in "${MODELS_FOR_THIS_LANG[@]}"
        do
            LOWER_CASE_ARG=""
            # Check if the term "uncased" is present anywhere in the model's name
            if [[ "$MODEL" =~ "uncased" ]]; then
                LOWER_CASE_ARG="--lower"
            fi
            CMD="$BASE_CMD $LOWER_CASE_ARG --pre_trained_model $MODEL --dataset $DATASET --train_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-train-$SHORT_LANGUAGE.tsv --test_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-test-$SHORT_LANGUAGE.tsv --dev_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-dev-$SHORT_LANGUAGE.tsv --language multilingual"
            echo "Executing: $CMD"
            eval $CMD
        done

        # Then handle the language-specific models
        KEY="$DATASET,$LANGUAGE"
        if [[ -n "${LANGUAGE_SPECIFIC_MODELS[$KEY]}" ]]; then
            IFS=',' read -ra SPECIFIC_MODELS_FOR_LANGUAGE <<< "${LANGUAGE_SPECIFIC_MODELS[$KEY]}"
            for MODEL in "${SPECIFIC_MODELS_FOR_LANGUAGE[@]}"
            do
                LOWER_CASE_ARG=""
                # Check if the term "uncased" is present anywhere in the model's name
                if [[ "$MODEL" =~ "uncased" ]]; then
                    LOWER_CASE_ARG="--lower"
                fi
                CMD="$BASE_CMD $LOWER_CASE_ARG --pre_trained_model $MODEL --dataset $DATASET --train_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-train-$SHORT_LANGUAGE.tsv --test_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-test-$SHORT_LANGUAGE.tsv --dev_dataset ../data/$DATASET/$SHORT_LANGUAGE/HIPE-2022-v2.1-$DATASET-dev-$SHORT_LANGUAGE.tsv --language $LANGUAGE"
                echo "Executing: $CMD"
                eval $CMD
            done
        fi
    done
done

