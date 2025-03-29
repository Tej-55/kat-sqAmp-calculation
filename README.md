# TITANS for Squared Amplitude Calculation

This repository implements and evaluates the Memory As Context (MAC) Transformer architecture from Google's [Titans: Learning to Memorize at Test Time](https://arxiv.org/pdf/2501.00663) paper for calculating squared amplitudes in particle physics. The project compares the performance of a standard T5 transformer model against the more advanced TITANS architecture on this specialized task.

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

### 2. MAC TITANS Model
The Memory As Context Transformer architecture with:
- Neural memory components for enhanced pattern recognition
- Segmented attention for handling long sequences
- Adaptive memory updates for learning complex transformations

## Results

### Performance Comparison

| Model | Test Sequence Accuracy | Test Token Accuracy |
|-------|------------------------|---------------------|
| T5 Transformer | 41.77% | 56.34% |
| MAC TITANS | 60.54% | 71.55% |

The TITANS model significantly outperforms the baseline T5 transformer, demonstrating its effectiveness for this task.

### Training Curves

<div align="center">
  <img src="t5_output\training_history.png" width="45%" alt="T5 Training Loss"/>
  <img src="titans_output\training_history.png" width="45%" alt="TITANS Training Loss"/>
</div>
<p align="center"><i>Left: T5 Training Loss, Right: TITANS Training Loss</i></p>

## Repository Structure

```
├── data/                  # Data files and preprocessing scripts
├── t5_output/             # T5 model outputs and checkpoints
├── titans_output/         # TITANS model outputs and checkpoints
├── titans/                # TITANS architecture implementation
│   ├── memory_models.py   # Memory model implementations
│   ├── neural_memory.py   # Neural memory components
│   ├── mac_transformer.py # Memory As Context Transformer
│   └── associative_scan.py # Associative scan operations
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data loading and processing
│   ├── tokenizer.py       # Custom tokenizer implementation
│   └── train_utils.py     # Training and evaluation functions
├── t5_main.py             # T5 model training script
├── titans_main.py         # TITANS model training script
└── README.md              # This file
```

## Usage

### Model Weights

The trained model weights are too large to be hosted on GitHub. You can download them from the following Google Drive location:

[Download Model Weights](https://drive.google.com/drive/folders/1KDbJDqwSeiInFBvnL0iuImQRKYUG-kgV?usp=sharing)

The folder contains:
- `amplitude_model.pth` - T5 model weights
- `titans_model.pt` - MAC TITANS model weights

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

### Training TITANS Model
```bash
python titans_main.py \
  --batch_size=8 \
  --num_epochs=8 \
  --lr=5e-5 \
  --segment_len=512 \
  --dim=128 \
  --depth=6 \
  --heads=8 \
  --depth_mem=2
```

## Acknowledgements

This code is adapted from the following sources:
- TITANS-PyTorch by lucidrains: https://github.com/lucidrains/titans-pytorch
- Skanformer by Riteshbhalerao11 : https://github.com/Riteshbhalerao11/Skanformer

The project was developed as part of the Google Summer of Code application for the "TITANS for squared amplitude calculation" project.
