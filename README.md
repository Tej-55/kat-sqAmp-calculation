# Titans for Squared Amplitude Calculation

This repository implements and evaluates the Kolmogorov-Arnold Transformer for calculating squared amplitudes in particle physics. The project compares the performance of a standard T5 transformer model against the more advanced KAT architecture on this specialized task.

## Project Overview

In particle physics, squared amplitudes are crucial for calculating cross-sections, which provide a testable link between theory and experiment. This project explores how modern transformer architectures can learn to map from amplitudes to their squared forms using sequence-to-sequence modeling.

## Dataset

The [dataset](https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs) consists of particle physics expressions in the format:
```
"event type : Feynman diagram : amplitude : squared amplitude"
```
Amplitudes serve as input sequences, while squared amplitudes are the target outputs for the model.

## Data Preprocessing

The preprocessing of amplitude expressions involves several key steps to normalize indices and prepare the data for the model. Since cross-section calculations involve long symbolic expressions, proper formatting and normalization are crucial for effective training.

### Index Normalization

The code uses regular expressions to identify and normalize different types of indices present in symbolic expressions:

1. **Pattern for momentum indices**:
   ```python
   self.pattern_momentum = re.compile(r'\b[ijkl]_\d{1,}\b')
   ```
   This pattern identifies component indices like `i_36289`, `j_5`, `k_3`, or `l_36277`.

2. **Pattern for variable indices**:
   ```python
   self.pattern_num_123 = re.compile(r'\b(?![ps]_)\w+_\d{1,}\b')
   ```
   This matches variable indices like `%sigma_157721` or `%gam_166722` while explicitly excluding momentum indices (`p_1`) and Mandelstam variables (`s_13`).

3. **Pattern for compound indices**:
   ```python
   self.pattern_index = re.compile(r'\b\w+_\w+_\d{2,}\b')
   ```
   This targets compound identifiers with two underscores where the final part is a numeric value with at least two digits, like `e_eps_207381`. However, I did not encounter such patterns in this dataset.

### Normalization Process

The normalization process replaces large, arbitrary numerical indices (e.g., `_157721`) with sequential small indices (`_1`, `_2`, etc.), maintaining consistency across expressions. This approach mirrors MARTY’s indexing mechanism, where Greek-lettered variables cycle through different numerical values.

A fixed pool of tokens is created for both index and momentum terms, ensuring a structured replacement during tokenization. This reduces vocabulary size while preserving the integrity of expressions. Unlike amplitudes, squared amplitudes retain their original structure, as they do not contain dynamically generated indexed terms requiring normalization. Instead, they undergo a simpler tokenization process that isolates operators, variables, and structured identifiers using regular expressions.

## Models Implemented

### 1. T5 Transformer (Baseline)
A standard encoder-decoder transformer model using the Google's [T5 Small](https://huggingface.co/google-t5/t5-small) model.

### 2. KAT
GR-KAN is added instead of the regular feedforward network in the T5's architecture.

## Results

### Performance Comparison

| Model | Test Sequence Accuracy | Test Token Accuracy |
|-------|------------------------|---------------------|
| T5 Transformer | 41.77% | 56.34% |
| T5 KAT | 48.59% | 52.68% |

The KAT model significantly outperforms the baseline T5 transformer, demonstrating its effectiveness for this task.

### Training Curves

<div align="center">
  <img src="t5_output\training_history.png" width="45%" alt="T5 Training Loss"/>
  <img src="kat_output\training_history.png" width="45%" alt="KAT Training Loss"/>
</div>
<p align="center"><i>Left: T5 Training Loss, Right: KAT Training Loss</i></p>

## Repository Structure

```
├── data/                  # Data files and preprocessing scripts
├── t5_output/             # T5 model outputs and checkpoints
├── KAT_output/            # KAT model outputs and checkpoints
├── KAT/                # KAT architecture implementation
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data loading and processing
│   ├── tokenizer.py       # Custom tokenizer implementation
│   └── train_utils.py     # Training and evaluation functions
├── t5_main.py             # T5 model training script
└── README.md              # This file
```

## Usage

### Model Weights

The trained model weights are too large to be hosted on GitHub. You can download them from the following Google Drive location:

[Download Model Weights](https://drive.google.com/drive/folders/1KDbJDqwSeiInFBvnL0iuImQRKYUG-kgV?usp=sharing)

### Prerequisites
```bash
pip install tensordict einx axial_positional_embedding rotary_embedding_torch x_transformers hyper_connections accelerated_scan
pip install torchtext==0.17.2 torch torchvision
```

### Training T5 Baseline
```bash
python t5_main.py \
  --batch_size=8 \
  --num_epochs=8 \
  --lr=5e-5
```

### Training KAT Model
```bash
python t5_main.py \
  --kat-mode=True \
  --batch_size=8 \
  --num_epochs=8 \
  --lr=5e-5
```

## Acknowledgements

This code is adapted from the following sources:
- KAT by Adamdad: https://github.com/Adamdad/kat
- Skanformer by Riteshbhalerao11 : https://github.com/Riteshbhalerao11/Skanformer

The project was developed as part of the Google Summer of Code application for the "Titans for squared amplitude calculation" project.
