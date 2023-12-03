import random
import h5py
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import math

def slideWindow(a, size, step):
    corr_size = list(a.shape)
    corr_size[0] = math.ceil(corr_size[0] / step) * step
    c = torch.zeros(corr_size)
    c[:len(a)] = a
    return c.unfold(dimension=0, size=size, step=step)
    
def prepareSet(group, labels, patch_len, patch_skip, max_seqs=None, min_seqs=None, single_first=False):
    X = []
    Y = []
    
    sets = group.keys()
    sets = sorted(sets, key=lambda x: x.count(','), reverse=(not single_first))
    spec_count = torch.zeros(len(labels))
    for species in tqdm(list(sets)):
        if max_seqs is not None:
            signal = group.get(species)[:max_seqs * patch_skip]
        else:
            signal = group.get(species)
        signal = torch.as_tensor(np.asarray(signal))
        label = torch.zeros(len(labels))
        for s in species.split(','):
            if s in labels:
                label[labels[s]] = 1
        if max_seqs is not None and min_seqs is not None:
            ma = torch.all(spec_count[torch.nonzero(label)] < max_seqs) # all labels have less then max
            mi = torch.any(spec_count[torch.nonzero(label)] < min_seqs) # one label has less than min
            if label.sum() > 0 and len(signal) > 0 and (mi or ma):
                patches = slideWindow(signal, patch_len, patch_skip)
                X.extend(patches)
                Y.extend([label] * len(patches))
                spec_count += label * len(patches)
        else:
                patches = slideWindow(signal, patch_len, patch_skip)
                X.extend(patches)
                Y.extend([label] * len(patches))
            
    
    X, Y = shuffle(X, Y, random_state=42)
    return torch.stack(X), torch.stack(Y)

def prepare(file, labels, patch_len, patch_skip, max_seqs=None, min_seqs=None, single_first=False):
    prepared_hf = h5py.File(file, 'r')
    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, patch_skip, max_seqs, min_seqs, single_first)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def rand_y(Y, exclude_class):
    for _ in range(len(Y)):
        idx = random.randint(0, len(Y)-1)
        if torch.logical_and(Y[idx], exclude_class).sum() == 0:
            return idx
    return -1

# X and Y need to be shuffled
def mixup(X, Y, min_seq=2, max_seq=2, p_min=1.0, p_max=1.0):
    X2 = X.clone()
    Y2 = Y.clone()
    for i, y in enumerate(Y):
        rand_k = random.randint(min_seq, max_seq)
        for k in range(rand_k-1):
            idx = rand_y(Y, Y2[i])
            if idx != -1:
                p = random.uniform(p_min, p_max)
                X2[i] += p * X[idx]
                Y2[i] += Y[idx]
        X2[i] /= rand_k
    return X2, Y2

def rand_y2(Y, exclude_class, k):
    for _ in range(len(Y)):
        idx = random.randint(0, len(Y)-1)
        if torch.logical_and(Y[idx], exclude_class).sum() == 0 and Y[idx].sum() < k:
            return idx
    return -1

# X and Y need to be shuffled, but with max k
def mixup2(X, Y, min_seq=2, max_seq=2, p_min=1.0, p_max=1.0):
    X2 = X.clone()
    Y2 = Y.clone()
    for i, y in enumerate(Y):
        k = random.randint(min_seq, max_seq) # how many species in one seq
        k -= Y2[i].sum() # minus the ones already there
        l = 1
        while k > 0:
            idx = rand_y(Y, Y2[i], k) # find other species, at maximum k
            if idx != -1:
                p = random.uniform(p_min, p_max)
                X2[i] += p * X[idx]
                Y2[i] += Y[idx]
                k -= Y[idx].sum()
                l += 1
            else:
                k = 0
        X2[i] /= l
    return X2, Y2

def preprocess(x, n_fft=512):
    x = torch.abs(torch.stft(x, n_fft=n_fft, window=torch.hann_window(window_length=n_fft).to(x.device), return_complex=True)) # FFT
    #x = 20 * torch.log10(x / torch.max(x) + 1e-10) # amplitude to db
    
    x = 20.0 * torch.log10(torch.clamp(x, min=1e-10))
    x -= 20.0 * torch.log10(torch.clamp(torch.max(x), min=1e-10))
    
    x = torch.abs(x - x.mean(dim=2, keepdim=True).repeat((1, 1, x.shape[2]))) # noise filter
    x /= x.amax(1, keepdim=True).amax(2, keepdim=True) # normalize spectrograms between 0 and 1
    return x.transpose(2, 1)

def noise(x, std=0.05):
    x += std * torch.randn(x.shape).to(x.device)
    return x

def getCorrects(output, target, threshold=0.5):
    log_and = torch.logical_and(output > threshold, target > threshold)
    corr = 0.0
    for i, t in enumerate(target):
        corr += log_and[i].sum() / max((t > threshold).sum(), (output[i] > threshold).sum())
    return corr

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)