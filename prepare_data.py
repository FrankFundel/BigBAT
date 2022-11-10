import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from scipy import signal
from tqdm import tqdm
import h5py
import sys
import argparse

parser = argparse.ArgumentParser(description='Preparing data.')

parser.add_argument('--run', help='Use this flag to run the prepare loop.', action="store_true")
parser.add_argument('--sample_rate', type=int, help='Desired sample rate.', default=22050)
parser.add_argument('--n_fft', type=int, help='Desired number of fft.', default=512)
parser.add_argument('--filename', help='Desired filename to write to.', default="prepared_signal.h5")

args, unknown = parser.parse_known_args()

sample_rate = args.sample_rate      # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz)
n_fft = args.n_fft                  # 23 ms * 22050 Hz ~ 512

# Smaller values improve the temporal resolution of the STFT at the expense of frequency resolution
# Shape: (1+nfft/2, n_frames = len/(n_fft/4))

# 1th order butterworth high-pass filter with cut-off frequency of 15,000 kHz
b, a = signal.butter(10, 15000 / 120000, 'highpass')

def prepareData(y):
    filtered = signal.lfilter(b, a, y)                      # filter
    return filtered

def mergeClass(name):
    signals = []
    for filename in tqdm(classes[name]):
        y, _ = librosa.load("../data/" + filename + '.wav', sr=sample_rate)
        sig = prepareData(y)
        signals.append(sig)
        
    if len(signals) >= 7:
        X_train, X_test, _, _ = train_test_split(signals, np.zeros(len(signals)), test_size=0.25, random_state=42)
        X_train, X_val, _, _ = train_test_split(X_train, np.zeros(len(X_train)), test_size=0.2, random_state=42)
        train = np.concatenate(X_train)
        test = np.concatenate(X_test)
        val = np.concatenate(X_val)
        train_set.create_dataset(name, data=train)
        test_set.create_dataset(name, data=test)
        val_set.create_dataset(name, data=val)

if args.run:
    df = pd.read_csv('../data.csv')
    hf = h5py.File(args.filename, 'a')  # will be ~13GB
    train_set = hf.require_group("train")
    test_set = hf.require_group("test")
    val_set = hf.require_group("val")
    
    classes = {
        "Rhinolophus ferrumequinum": [],
        "Rhinolophus hipposideros": [],
        "Myotis daubentonii": [],
        "Myotis brandtii": [],
        "Myotis mystacinus": [],
        "Myotis emarginatus": [],
        "Myotis nattereri": [],
        "Myotis bechsteinii": [],
        "Myotis myotis": [],
        "Nyctalus noctula": [],
        "Nyctalus leisleri": [],
        "Nyctalus lasiopterus": [],
        "Pipistrellus pipistrellus": [],
        "Pipistrellus pygmaeus": [],
        "Pipistrellus nathusii": [],
        "Pipistrellus kuhlii": [],
        "Hypsugo savii": [],
        "Vespertilio murinus": [],
        "Eptesicus serotinus": [],
        "Eptesicus nilssonii": [],
        "Plecotus auritus": [],
        "Plecotus austriacus": [],
        "Barbastella barbastellus": [],
        "Tadarida teniotis": [],
        "Miniopterus schreibersii": [],
        "Myotis capaccinii": [],
        "Myotis dasycneme": [],
        "Pipistrellus maderensis": [],
        "Rhinolophus blasii": []
    }

    for index, row in df.iterrows():
        classes[row["species"]].append(row["filename"])

    print("sorted!")
    
    for classname in list(classes):
        if not classname in train_set or not classname in test_set or not classname in val_set:
            mergeClass(classname)
            print(classname + " prepared!")

    hf.close()