# BigBAT
Enhancement on the previous BioAcousticTransformer, including unsupervised pre-training, augmentation, species label smoothing and meta data.

# Content
- ASL.py: AsymmetricLoss Function
- Transformer.py: ViT Architecture
- SAM.py: Sharpness Aware Minimization Optimizer
- tools.py: functions for dataset loading and preprocessing
- BigBAT.py: The BigBAT model
- BigBAT_\[model\].py: The BigBAT model, modified for \[model\]

# Notice
I moved the experiment files in directories, references to files and tools are not valid anymore and need to be fixed.

# Tutorial
For the experiments, h5 file with a specific format is necessary for fast dataloading and training.
You start off with a folder of .wav-files of arbitrary length and a meta.csv with filename and species for each file.

If you start off with a folder with .zc files with the labels which are not inside the meta.csv:
Run preparation/zcjs/dir_2_meta.js to create a meta_n.csv, where labels are extracted from .zc files (please edit the script accordingly)

Next, run preparation/prepare_data.py (please edit the script accordingly)
spec = row["ORGID"]
row["IN FILE"]
The first one is the species (comma seperated)
The second one is the path to the wav
The csv is seperated by semicolons
The csv needs those two columns, you can rename them if you need.

This results in a .h5 file with all recordings concatenated for each class that occurs in meta_n.csv, split into train, test and validation set.

To load the dataset, use:

nfft = 512
num_bands = nfft // 2 + 1

max_len = 60
patch_len = 44
patch_skip = 22

samples_per_step = patch_skip * (nfft // 4)
seq_len = (max_len + 1) * samples_per_step
seq_skip = seq_len // 4

max_seqs = 1000
min_seqs = 100

data_path = "prepared_signal.h5"
X_train, Y_train, X_test, Y_test, X_val, Y_val = prepare(data_path, classes, seq_len, seq_skip, max_seqs, min_seqs)
