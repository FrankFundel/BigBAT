# BigBAT

## Overview
Enhancement on the previous BioAcousticTransformer, including unsupervised pre-training, augmentation, species label smoothing and meta data.

---

# Content
- ASL.py: AsymmetricLoss Function
- Transformer.py: ViT Architecture
- SAM.py: Sharpness Aware Minimization Optimizer
- tools.py: functions for dataset loading and preprocessing
- BigBAT.py: The BigBAT model
- BigBAT_\[model\].py: The BigBAT model, modified for \[model\]
- train.py: Simple training script

---

## 1. Data Preparation

### Prerequisites
- Audio files in `.wav` format.
- A metadata file (`meta.csv`) including information about each audio file.

### Data Preparation Script Arguments
- `wav_folder`: Path to the folder containing `.wav` files.
  - **Type**: String
  - **Description**: Directory where `.wav` files are stored.
- `meta_file`: Path to the metadata file.
  - **Type**: String
  - **Description**: Path to your metadata file (`meta.csv`).
- `--sample_rate`: Desired sample rate for audio files.
  - **Type**: Integer
  - **Default**: 22050
  - **Description**: Sample rate for audio conversion.
- `--n_fft`: Number of FFT points.
  - **Type**: Integer
  - **Default**: 512
  - **Description**: FFT points for audio processing.
- `--output_file`: Filename for the output file.
  - **Type**: String
  - **Default**: "prepared.h5"
  - **Description**: Filename for the processed `.h5` file.

### Running the Data Preparation Script
- Configure and run the script to process `.wav` files and metadata into a `.h5` file.

---

## 2. Training

### Training Script Arguments
- `json_file`: Path to JSON file with class names.
- `--nfft`: Number of FFT points (default: 512).
- `--max_len`: Maximum sequence length (default: 60).
- `--patch_len`: Length of each patch (default: 44).
- `--patch_skip`: Skip length for patches (default: 22).
- `--max_seqs`: Maximum number of sequences (optional).
- `--min_seqs`: Minimum number of sequences (optional).
- `data_path`: Path to the `.h5` data file.
- `--holdout`: Data proportion for holdout (0 to 1).
- `--batch_size`: Batch size (default: 128).
- `--epochs`: Number of training epochs (default: 15).
- `--lr`: Learning rate (default: 0.001).
- `--warmup_epochs`: Warm-up epochs (default: 3).
- `--wandb_project`: Weights & Biases project name.
- `--wandb_entity`: Weights & Biases entity name.
- `--model_filename`: Filename for the saved model (default: 'BigBAT.pth').
- `--repeats`: Number of repeats for evaluation (default: 5).
- `--figure_filename`: Filename for saving figures (default: 'confusion_matrix.png').
- `--method`: Training method (choices: 'standard', 'BYOL', 'FixMatch', 'PseudoLabel').
- `--T1`: Epoch to introduce unlabeled data.
- `--T2`: Epoch to unintroduce unlabeled data.
- `--every_n`: Frequency for using unlabeled data in batches.
- `--no_mixup`: Flag to disable mixup.
- `--lambda_u`: Weight of unlabeled loss in FixMatch.

### Running the Training Script
- Set the arguments based on your dataset and requirements.
- Execute the training script to train your model.
- Monitor progress through console logs and optional Weights & Biases integration.

### Output
- A trained model (`BigBAT.pth`).
