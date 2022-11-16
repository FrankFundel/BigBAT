# BigBAT
Enhancement on the previous BioAcousticTransformer, including unsupervised pre-training, augmentation, manual label smoothing, GPS and time data.

# To Do
- Augmentation (noise, masking)
    - SpecAugment: https://arxiv.org/abs/1904.08779
- Species-Label-Smoothing
- Prepare data from UFS
- Semi-supervised pretraining
    - DINO: https://arxiv.org/pdf/2104.14294
    - SiT: https://arxiv.org/abs/2104.03602
    - SSAST: https://arxiv.org/abs/2110.09784
    - Auto-encoder: https://www.ni.tu-berlin.de/fileadmin/fg215/teaching/nnproject/cnn_pre_trainin_paper.pdf
- Use location, time, altitude, weather, geo data (multi-branch or seperate networks)
    - GPS2Vec: https://yifangyin.github.io/publications/TMM-21.pdf
    - Time2Vec: https://arxiv.org/abs/1907.05321
- Create GUI
- Try other datasets
