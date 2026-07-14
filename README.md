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
