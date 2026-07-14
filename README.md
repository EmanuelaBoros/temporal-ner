# Temporal NER

[![Paper](https://img.shields.io/badge/arXiv-2606.27881-b31b1b.svg)](https://arxiv.org/abs/2606.27881)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-NER-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Temporal fusion strategies for named entity recognition in historical texts.**

This repository contains the code, data-processing utilities, experiments, and analysis material associated with:

> **A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts**  
> Emanuela Boros  
> [arXiv:2606.27881](https://arxiv.org/abs/2606.27881)

Historical named entities are not temporally stable: names, spellings, organizations, products, locations, and their relative prominence change over time. This project studies whether document dates can be integrated directly into Transformer-based NER models to improve robustness across historical periods.

<p align="center">
  <img src="temporal-ner/images/yearly_f1_by_strategy_type.png" alt="Yearly NER performance by temporal fusion strategy" width="850">
</p>

## Overview

The implementation extends multilingual Transformer token-classification models with lightweight temporal representations. It supports:

- **absolute and relative year encodings**;
- **early and late temporal fusion**;
- additive, concatenation, FiLM, adapter, multiscale, and cross-attention mechanisms;
- multilingual historical NER experiments on **French and German HIPE data**;
- token-level evaluation, per-entity analysis, yearly analysis, and temporal probing;
- optional Weights & Biases logging and distributed training utilities.

The experiments described in the paper show that temporal information can be useful, but its effect depends strongly on where and how it is injected. Late-fusion approaches are generally more robust, particularly for earlier and noisier historical periods.

## Repository structure

```text
.
├── temporal-ner/                 # Main temporal NER implementation
│   ├── main.py                   # Training and evaluation entry point
│   ├── models.py                 # NER models and temporal fusion modules
│   ├── dataset.py                # HIPE/CoNLL reader and label alignment
│   ├── probe.py                  # Temporal probing experiments
│   ├── postprocess.py            # Result aggregation and post-processing
│   ├── plots.py                  # Plotting utilities
│   ├── data/hipe2020/            # French, German, and English HIPE files
│   ├── experiments/              # Saved experiment metadata and statistics
│   ├── images/                   # Figures generated from the experiments
│   └── notebooks/                # Result-analysis notebooks
├── simple-ner/                   # Simpler NER baselines and prototypes
├── stacked-ner/                  # Stacked embedding NER experiments
├── old-stacked-ner/              # Some random stacked NER experiments
├── hf/                           # Hugging Face model configuration/code
├── HIPE-scorer_backup/           # Local copy of the HIPE evaluation utilities
├── results/                      # Selected result files
├── probing_results.tsv           # Aggregated probing results
└── LICENSE                       # GNU General Public License v3.0
```

The actively relevant implementation for the paper is under [`temporal-ner/`](temporal-ner/).

## Installation

A recent Python 3 environment is recommended. The repository does not currently pin package versions, so create an isolated environment before installing the dependencies.

```bash
git clone https://github.com/EmanuelaBoros/temporal-ner.git
cd temporal-ner

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install \
  torch \
  transformers \
  accelerate \
  peft \
  seqeval \
  pandas \
  numpy \
  tqdm \
  wandb \
  tensorboard \
  scikit-learn \
  matplotlib \
  jupyter
```

For GPU training, install the PyTorch build matching the CUDA version available on your system.

## Data format

The main loader expects HIPE-style tab-separated data with token-level NER annotations and document-level date metadata. Dates are read from lines such as:

```text
# hipe2022:date = 1888-01-09
```

The loader currently trains on the coarse literal NER task:

```text
NE-COARSE-LIT
```

The expected columns follow the HIPE format, including fields such as `TOKEN`, `NE-COARSE-LIT`, `NE-FINE-LIT`, `NE-NESTED`, and NEL-related annotations. Sequences are split on blank lines or `EndOfSentence` markers.

Example files are organized by language:

```text
temporal-ner/data/hipe2020/
├── de/
│   ├── HIPE-2022-v2.1-hipe2020-train-de.tsv
│   ├── HIPE-2022-v2.1-hipe2020-dev-de.tsv
│   └── HIPE-2022-v2.1-hipe2020-test-de.tsv
└── fr/
    ├── HIPE-2022-v2.1-hipe2020-train-fr.tsv
    ├── HIPE-2022-v2.1-hipe2020-dev-fr.tsv
    └── HIPE-2022-v2.1-hipe2020-test-fr.tsv
```

Please consult and respect the licenses and citation requirements of the original HIPE datasets.

## Temporal representations

The model can use the year either directly or as an offset from the minimum supported year.

| Representation | Command-line setting | Description |
|---|---|---|
| Absolute | default | Uses the calendar year as the temporal index. |
| Relative | `--use_relative_year` | Maps a year to a zero-based offset from the minimum year. |

The current dataset implementation assumes dates between **1700 and 2025**.

## Fusion strategies

Pass a strategy using `--temporal_fusion_strategy`.

| Strategy | Type | Description |
|---|---|---|
| `baseline` | No temporal fusion | Uses only the contextual token representations. |
| `add` | Late fusion | Adds a learned year embedding to every token. |
| `concat` | Late fusion | Concatenates token and year representations, followed by projection. |
| `film` | Late fusion | Applies feature-wise affine modulation conditioned on the year. |
| `adapter` | Late fusion | Adds a small year-conditioned feed-forward adapter. |
| `relative` | Late fusion | Encodes the temporal representation before FiLM-style modulation. |
| `multiscale` | Late fusion | Combines year, decade, and century embeddings. |
| `early-cross-attention` | Early fusion | Applies token-to-time cross-attention before the Transformer encoder. |
| `late-cross-attention` | Late fusion | Applies token-to-time cross-attention after contextual encoding. |

## Quick start

Run commands from the main implementation directory:

```bash
cd temporal-ner
```

### 1. Baseline model

```bash
python main.py \
  --model_name_or_path dbmdz/bert-base-historic-multilingual-cased \
  --model_type time \
  --temporal_fusion_strategy baseline \
  --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
  --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
  --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --max_sequence_len 512 \
  --epochs 5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --output_dir experiments \
  --device cuda \
  --do_train
```

### 2. Late cross-attention with relative years

```bash
python main.py \
  --model_name_or_path dbmdz/bert-base-historic-multilingual-cased \
  --model_type time \
  --temporal_fusion_strategy late-cross-attention \
  --use_relative_year \
  --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
  --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
  --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --max_sequence_len 512 \
  --epochs 5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --output_dir experiments \
  --device cuda \
  --do_train
```

### 3. German experiment

Replace the three French paths with the corresponding files under `data/hipe2020/de/`:

```bash
python main.py \
  --model_name_or_path dbmdz/bert-base-historic-multilingual-cased \
  --model_type time \
  --temporal_fusion_strategy film \
  --use_relative_year \
  --train_dataset data/hipe2020/de/HIPE-2022-v2.1-hipe2020-train-de.tsv \
  --dev_dataset data/hipe2020/de/HIPE-2022-v2.1-hipe2020-dev-de.tsv \
  --test_dataset data/hipe2020/de/HIPE-2022-v2.1-hipe2020-test-de.tsv \
  --max_sequence_len 512 \
  --epochs 5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --output_dir experiments \
  --device cuda \
  --do_train
```

Use `--device cpu` for a CPU-only run. Training will be substantially slower.

## Evaluation

To evaluate a saved checkpoint, provide the same model, data, temporal settings, and the checkpoint directory:

```bash
python main.py \
  --model_name_or_path dbmdz/bert-base-historic-multilingual-cased \
  --model_type time \
  --temporal_fusion_strategy late-cross-attention \
  --use_relative_year \
  --train_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
  --dev_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv \
  --test_dataset data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --max_sequence_len 512 \
  --output_dir evaluation \
  --checkpoint PATH/TO/CHECKPOINT \
  --device cuda \
  --do_eval
```

The evaluation code reports sequence-labeling precision, recall, and F1 using `seqeval`, and writes prediction and analysis artifacts to the experiment directory.

## Experiment outputs

Each run creates a model-specific directory under `--output_dir`. Depending on the selected mode, it may contain:

```text
logging.log
label_map.json
entity_statistics_train.tsv
entity_statistics_dev.tsv
entity_statistics_test.tsv
pytorch_model.bin
all_results_*.json
prediction files
```

The output directory name records the model, maximum sequence length, number of epochs, model type, fusion strategy, and temporal representation.

## Optional experiment tracking

Enable Weights & Biases logging with:

```bash
--wandb
```

Runs are logged under the project name `long-horizon` by the current implementation.

## Analysis and visualisation

The repository contains notebooks and generated figures for:

- yearly F1 by temporal strategy;
- absolute versus relative temporal encoding;
- entity-frequency and entity-type analyses;
- gains relative to the non-temporal baseline;
- early versus late fusion comparisons;
- temporal probing accuracy;
- representation visualisation with t-SNE.

Start Jupyter from the repository root:

```bash
jupyter notebook temporal-ner/notebooks/
```

## Reproducibility notes

- Set the random seed with `--seed`; the default is `42`.
- Keep the dataset split, tokenizer, label map, and temporal representation identical when comparing fusion strategies.
- The code automatically generates a `label_map.json` inside the experiment directory.
- Experiments may be skipped when result files already exist in the target directory.
- For fair temporal comparisons, use the same encoder, sequence length, learning rate, batch size, and number of epochs.
- This is research code and contains several experimental branches; inspect the selected configuration before launching large runs.

## Citation

Please cite the paper when using this repository:

```bibtex
@article{boros2026temporalner,
  title   = {A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts},
  author  = {Boros, Emanuela},
  journal = {arXiv preprint arXiv:2606.27881},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.27881}
}
```

## License

The code is released under the [GNU General Public License v3.0](LICENSE).

Datasets, pretrained models, and third-party evaluation tools remain subject to their respective licenses and terms of use.




