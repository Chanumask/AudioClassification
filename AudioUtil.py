import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from torchaudio import transforms
import torchaudio

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
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def get_spectrogram(aud, n_mels=NMELS, n_fft=NFFT, hop_len=HOPLENGTH):
        signal, sr = aud
        top_db = TOPDB

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = t.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        #  Time Masking with param possible length of the mask sampled uniformly from [0, time_mask_param)
        # if TIME_MASKING:
        #     spec = t.TimeMasking(5)(spec)

        # Frequency Masking with param possible length of the mask sampled uniformly from [0, frequency_mask_param)
        # if FREQ_MASKING:
        #     spec = t.FrequencyMasking(2)(spec)

        # Convert to decibels
        # spec = (transforms.AmplitudeToDB(top_db=top_db, stype="power")(spec) + 40) / 40
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        # a2b = transforms.AmplitudeToDB(top_db=top_db, stype="power")
        # spec = a2b(spec)
        # print(a2b.)
        # print(spec.min(), spec.max(), spec.mean())

        return spec


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


class Mixup(nn.Module):
    def __init__(self, alpha, num_classes):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.beta_dist = Uniform(alpha, 1)  # if uniform use values from 0.5 upwards or Beta(alpha, alpa)

    @classmethod
    def mix(cls, data: torch.Tensor, gamma, indices: torch.Tensor) -> torch.Tensor:
        assert data.shape[0] == indices.shape[0], 'Requires same number of samples'
        gamma_ = gamma.view(gamma.shape[0], *(1 for _ in range(len(data.shape) - 1)))
        return data * gamma_ + (1.0 - gamma_) * data[indices]

    def forward(self, data, labels):
        gamma = self.beta_dist.sample((data.shape[0],))
        indices = torch.randperm(data.size(0), device=data.device, dtype=torch.long)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes) if len(labels.shape) <= 1 else labels
        mixedup_data, mixedup_labels = self.mix(data, gamma, indices), self.mix(one_hot_labels, gamma, indices)
        return mixedup_data, mixedup_labels


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


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
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


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


class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        return torch.matmul(attn, x)
