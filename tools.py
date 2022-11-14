import random
import h5py
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import cv2
import torch

def slideWindow(a, size, step):
    b = []
    i = 0
    pos = 0
    while pos + size < len(a):
        pos = int(i * step)
        tile = a[pos : pos + size]
        b.append(tile)
        i+=1
    return b
    
def prepareSet(prepared_set, labels, patch_len, patch_skip):
    X = []
    Y = []

    for species in tqdm(list(labels)):
        signal = np.asarray(prepared_set.get(species))
        label = labels[species]

        patches = slideWindow(signal, patch_len, patch_skip)[:-1] # last one is not full
        X.extend(patches)
        Y.extend([label] * len(patches))
    
    X, Y = shuffle(X, Y, random_state=42)
    return np.asarray(X), np.asarray(Y)

def prepare(file, labels, patch_len, patch_skip):
    prepared_hf = h5py.File(file, 'r')
    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, patch_skip)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, patch_skip)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def rand_y(Y, exclude_class):
    while True:
        idx = random.randint(0, len(Y)-1)
        if torch.logical_and(Y[idx], exclude_class).sum() == 0:
            return idx

# X and Y need to be shuffled
def mixup(X, Y, num_classes, min_seq=2, max_seq=2, p_min=1.0, p_max=1.0):
    Y1 = one_hot(Y, num_classes, device=X.device)
    X2 = X.clone()
    Y2 = Y1.clone()
    for i, y in enumerate(Y):
        rand_k = random.randint(min_seq, max_seq)
        for k in range(rand_k-1):
            idx = rand_y(Y1, Y2[i])
            p = random.uniform(p_min, p_max)
            X2[i] += p * X[idx]
            Y2[i] += Y1[idx]
        X2[i] /= rand_k
    return X2, Y2

def preprocess(x, n_fft=512):
    x = torch.abs(torch.stft(x, n_fft=n_fft, window=torch.hann_window(window_length=n_fft).to(x.device), return_complex=True)) # FFT
    x = 20 * torch.log10(x / torch.max(x) + 1e-10) # amplitude to db
    x = torch.abs(x - x.mean(dim=2, keepdim=True).repeat((1, 1, x.shape[2]))) # noise filter
    x = x.transpose(dim0=2, dim1=1)
    x /= x.amax(1, keepdim=True).amax(2, keepdim=True) # normalize spectrograms between 0 and 1
    return x

def noise(x, std=0.05):
    x += std * torch.randn(x.shape).to(x.device)
    return x

def getCorrects(output, target, threshold=0.5):
    log_and = torch.logical_and(output > threshold, target > threshold)
    corr = 0.0
    for i, t in enumerate(target):
        corr += log_and[i].sum() / max((t > threshold).sum(), (output[i] > threshold).sum())
    return corr