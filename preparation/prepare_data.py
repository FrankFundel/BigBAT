import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from scipy import signal
from tqdm import tqdm
import h5py
import argparse

parser = argparse.ArgumentParser(description='Preparing data.')

parser.add_argument('wav_folder', type=str, help="Path to the wav folder.")
parser.add_argument('meta_file', type=str, help="Path to the meta file.")
parser.add_argument('--sample_rate', type=int, help='Desired sample rate.', default=22050)
parser.add_argument('--n_fft', type=int, help='Desired number of fft.', default=512)
parser.add_argument('--output_file', help='Desired filename to write to.', default="prepared.h5")

args, unknown = parser.parse_known_args()

wav_folder = args.wav_folder
meta_file = args.meta_file
sample_rate = args.sample_rate      # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz)
n_fft = args.n_fft                  # 23 ms * 22050 Hz ~ 512
output_file = args.output_file

# Smaller values improve the temporal resolution of the STFT at the expense of frequency resolution
# Shape: (1+nfft/2, n_frames = len/(n_fft/4))

# 1th order butterworth high-pass filter with cut-off frequency of 15,000 kHz
b, a = signal.butter(10, 15000 / 120000, 'highpass')

print(sample_rate)
def prepareData(y):
    filtered = signal.lfilter(b, a, y)                      # filter
    return filtered

def mergeClass(name):
    signals = []
    for filename in tqdm(classes[name]):
        y, _ = librosa.load(os.path.join(wav_folder, filename), sr=sample_rate)
        #y = prepareData(y)
        signals.append(y)
        
    if len(signals) >= 7:
        X_train, X_test, _, _ = train_test_split(signals, np.zeros(len(signals)), test_size=0.25, random_state=42)
        X_train, X_val, _, _ = train_test_split(X_train, np.zeros(len(X_train)), test_size=0.2, random_state=42)
        train = np.concatenate(X_train)
        test = np.concatenate(X_test)
        val = np.concatenate(X_val)
        train_set.create_dataset(name, data=train)
        test_set.create_dataset(name, data=test)
        val_set.create_dataset(name, data=val)

if __name__ == '__main__':
    df = pd.read_csv(meta_file, delimiter=';')
    hf = h5py.File(output_file, 'a')
    train_set = hf.require_group("train")
    test_set = hf.require_group("test")
    val_set = hf.require_group("val")

    classes = {}

    for index, row in df.iterrows():
        spec = str(row["MANUAL ID"])
        if spec not in classes:
            classes[spec] = []
        classes[spec].append(row["IN FILE"])

    print("sorted!")
    print(classes.keys())

    for classname in list(classes):
        classname = str(classname)
        if not classname in train_set or not classname in test_set or not classname in val_set:
            mergeClass(classname)
            print(classname + " prepared!")

    hf.close()