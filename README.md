# BigBAT
Enhancement on the previous BioAcousticTransformer, including unsupervised pre-training, augmentation, species label smoothing and meta data.

# Content
- ASL.py: AsymmetricLoss Function
- Transformer.py: ViT Architecture
- SAM.py: Sharpness Aware Minimization Optimizer
- tools.py: functions for dataset loading and preprocessing
- BigBAT.py: The BigBAT model
- BigBAT_\[model\].py: The BigBAT model, modified for \[model\]

# Tutorial

1. Data Preparation
A h5 file with a specific format is necessary for fast dataloading and training.
You start off with a folder of .wav-files of arbitrary length and a meta.csv with filename and species for each file.

If you start off with a folder with .zc files with the labels which are not inside the meta.csv:
Run preparation/zcjs/dir_2_meta.js to create a meta_n.csv, where labels are extracted from .zc files (please edit the script accordingly)

Next, run `preparation/prepare_data.py` (please edit the script accordingly)
- row["ORGID"]: the species name (comma seperated)
- row["IN FILE"]: the path to the wav
The csv is seperated by semicolons and needs those two columns, but you can rename them if you need.

This results in a .h5 file with all recordings concatenated for each class that occurs in meta_n.csv, split into train, test and validation set.

2. Training
