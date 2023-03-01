import os

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from AudioUtil import *
from settings import *
from tqdm import tqdm


class ESC50Data(Dataset):
    def __init__(self, mode, base, df, in_col, out_col, mixup):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i = {}
        self.i2c = {}
        self.categories = sorted(df[out_col].unique())
        self.mode = mode
        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category
        for ind in tqdm(range(len(df)), desc=self.mode + ' dataset', ncols=35):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[in_col])
            audio = AudioUtil.open(file_path)
            audio = AudioUtil.resample(audio, SAMPLE_RATE)
            audio = AudioUtil.rechannel(audio, 1)
            audio = AudioUtil.pad_trunc(audio, 4000)
            sgram = AudioUtil.get_spectrogram(audio, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            if mixup:
                mixer = MixupBYOLA(ratio=0.2, log_mixup_exp=True)
                sgram = mixer(sgram)
            self.data.append(sgram)
            label = (self.c2i[row['category']])
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx].shape)  # (1, 128, 431)
        return self.data[idx], self.labels[idx]