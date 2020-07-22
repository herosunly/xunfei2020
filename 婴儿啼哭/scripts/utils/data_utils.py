# coding=utf-8

import warnings
import librosa
import sklearn
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

class MyDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.mode = mode
        self.df = df

    def load_clip(self, filename):
        x, sr = librosa.load(filename)
        x += wgn(x, np.random.randint(1, 20))
        x = np.pad(x, (0, 728561 - x.shape[0]), 'constant')
        x = x[:sr*20] # 取前10秒
        return x, sr

    def extract_feature(self, filename):
        x, sr = self.load_clip(filename)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
        return mfcc
        # norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        # return norm_mfccs

    def __getitem__(self, index):
        wav_path = self.df['filename'].iloc[index]
        feature = self.extract_feature(wav_path)
        # print(feature.shape)
        if self.mode == 'test':
            label = torch.from_numpy(np.array(0))
        else:
            label = torch.from_numpy(np.array(self.df['label'].iloc[index]))
        return np.array(feature.reshape((1, 40, 862))), label

    def __len__(self):
        return len(self.df)
