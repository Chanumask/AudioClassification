from copy import deepcopy

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


def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()  # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[
                0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(
                    -1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + \
                           db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i + 1],
                                       band_bndry_freqs[i + 1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt

    else:
        return features
