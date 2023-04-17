import os
from copy import deepcopy

import librosa
import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from torchaudio import transforms
import torchaudio
from tqdm import tqdm

from settings import *
import torchaudio.transforms as t


class AudioUtil:

    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if sig.shape[0] == new_channel:
            # Nothing to do
            return aud
        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])
        return resig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            row_mean = torch.mean(sig, dim=1)

            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            # pad_begin = torch.stack([row_mean] * pad_begin_len, dim=1)
            # pad_end = torch.stack([row_mean] * pad_end_len, dim=1)

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def get_spectrogram(aud, specaug, mask_prob, n_mels=NMELS, n_fft=NFFT, hop_len=HOPLENGTH):
        signal, sr = aud
        top_db = TOPDB

        spec = t.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        # Time and Freq Masking with length of the mask sampled uniformly from [0, mask_param)
        random_float = random.random()
        if specaug and random_float < mask_prob:
            spec = t.TimeMasking(16)(spec)
            spec = t.FrequencyMasking(8)(spec)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        return spec


def calculate_esc_sample_length(dataset_name):
    print(f"calculating sample lengths of {dataset_name} dataset")
    if dataset_name == 'music':
        dir_path = "data/GTZAN/genres_original/blues"
    elif dataset_name == 'speech':
        dir_path = "data/SpeechCommands"
    elif dataset_name == 'primates':
        dir_path = "data/Primates/wav"
    elif dataset_name == 'esc50':
        dir_path = "data/ESC-50-master/audio"
    else:
        print('Invalid dataset name. Valid names are: music, speech, primates, esc50')
        return

    # Loop over all files in the directory and calculate the duration of each wav file
    durations = []
    for subdir, dirs, files in tqdm(os.walk(dir_path)):
        print(f"found {len(dirs)} subdirectories")
        for file_name in files:
            if file_name.endswith('.wav'):
                # Load the audio file
                file_path = os.path.join(subdir, file_name)
                audio, sr = librosa.load(file_path)

                duration = librosa.get_duration(y=audio, sr=sr)
                durations.append(duration)

    average_duration = np.mean(durations)
    std_duration = np.std(durations)

    print(f"Average duration: {average_duration:.2f} seconds")
    print(f"Standard deviation: {std_duration:.2f} seconds")


class RandomResizeCrop(nn.Module):
    def __init__(self, freq_scale=(0.75, 1.25), time_scale=(0.75, 1.25)):
        super().__init__()
        self.virtual_crop_scale = (1.0, 1.5)
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    def get_params(self, virtual_crop_size, in_size, num):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip((np.random.uniform(*self.freq_scale, num) * src_h).astype(int), 1, canvas_h)
        w = np.clip((np.random.uniform(*self.time_scale, num) * src_w).astype(int), 1, canvas_w)
        sh = canvas_h - h
        sw = canvas_w - w
        sw[sw <= 0], sh[sh <= 0] = 1, 1
        i = np.random.randint(0, sh, num)
        j = np.random.randint(0, sw, num)
        return i, j, h, w

    @staticmethod
    def crop_it(virtual_crop_area, i, x, y, h, w, shape):
        return F.interpolate(
            virtual_crop_area[None, i, :, x:x + h, y:y + w],
            size=shape,
            mode='bicubic',
            align_corners=True
        )

    def forward(self, lms):
        b, c, h, w = lms.shape
        new_h, new_w = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]  # new height and width
        virtual_crop_area = torch.zeros(b, c, new_h, new_w).float().to(lms.device)
        lh, lw = virtual_crop_area.shape[-2:]
        x, y = (lw - w) // 2, (lh - h) // 2  # compute center
        virtual_crop_area[:, :, y:y + h, x:x + w] = lms  # insert melspecs
        x, y, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], b)
        crop_futures = [torch.jit.fork(self.crop_it, virtual_crop_area, i, x, y, h, w, lms.shape[-2:])
                        for i, (x, y, h, w) in enumerate(zip(x, y, h, w))]
        return torch.cat([torch.jit.wait(fut) for fut in crop_futures], 0)


def log_mixup_exp(xa, xb, la, lb, alpha, n_classes):
    print("log mixup")
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    lmix = alpha * F.one_hot(la, n_classes) + (1. - alpha) * F.one_hot(lb, n_classes)
    return torch.log(x + torch.finfo(x.dtype).eps), lmix


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.
    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.2, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank_spec = []
        self.memory_bank_label = []

    def forward(self, x, y, num_classes):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank_spec:
            # get z as a mixing background sound
            zspec = self.memory_bank_spec[np.random.randint(len(self.memory_bank_spec))]
            zlabel = self.memory_bank_spec[np.random.randint(len(self.memory_bank_spec))]

            # mix them
            mixed_spec, mixed_labels = log_mixup_exp(x, zspec, y, zlabel, 1. - alpha, num_classes)
        else:
            mixed_spec = x
            mixed_labels = y
        # update memory bank
        self.memory_bank_spec = (self.memory_bank_spec + [x])[-self.n:]
        self.memory_bank_label = (self.memory_bank_label + [y])[-self.n:]

        return mixed_spec.to(torch.float), mixed_labels

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


def mixup(x, y, alpha=0.2):  # try alpha 1.0
    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha, size=x.size()[0])

    # Generate indices for the second audio sample
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # Load second audio sample and label
    x2 = x[index]
    y2 = y[index]

    # One-hot encode labels
    num_classes = y.max() + 1
    y_onehot = torch.zeros((batch_size, num_classes))
    y2_onehot = torch.zeros((batch_size, num_classes))
    for i in range(batch_size):
        y_onehot[i, y[i].long()] = 1
        y2_onehot[i, y2[i].long()] = 1

    # Create mixed audio inputs and labels
    mixed_x = []
    mixed_y = []
    for i in range(batch_size):
        lam_i = torch.from_numpy(np.array(lam[i])).reshape(1, 1, 1)
        mixed_x_i = lam_i * x[i] + (1 - lam_i) * x2[i]
        mixed_y_i = lam[i] * y_onehot[i] + (1 - lam[i]) * y2_onehot[i]
        mixed_x.append(mixed_x_i)
        mixed_y.append(mixed_y_i)

    mixed_x = torch.stack(mixed_x, dim=0)
    mixed_y = torch.stack(mixed_y, dim=0)

    return mixed_x, mixed_y


class RandomLinearFader(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

    def forward(self, lms):
        head, tail = self.gain * ((2.0 * np.random.rand(2)) - 1.0)  # gain * U(-1., 1) for two ends
        T = lms.shape[3]
        # print(lms.shape)
        slope = torch.linspace(head, tail, T, dtype=lms.dtype).reshape(1, 1, T).to(lms.device)
        y = lms + slope  # add liniear slope to log-scale input
        return y

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(gain={self.gain})'
        return format_string


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
