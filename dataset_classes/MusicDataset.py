import json
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from AudioUtil import *
from settings import *

SR = 22050  # 1 sec. of audio
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SR * TRACK_DURATION


def create_label_to_id_dict(dataset_path):
    """Creates a dictionary that maps each label to a unique integer ID."""
    labels = os.listdir(dataset_path)
    labels.sort()  # sort labels to ensure consistent ordering
    label_to_id = {label: i for i, label in enumerate(labels)}
    return label_to_id


def load_music_dataset(data_file_path):
    label_to_id = create_label_to_id_dict(MUSIC_DATASET_PATH)
    with open(data_file_path, "r") as f:
        file_paths = f.read().splitlines()

    labels = np.zeros(len(file_paths), dtype=int)

    for i, path in enumerate(file_paths):
        label = path.split("/")[0]
        label_id = int(label_to_id[label])
        labels[i] = label_id

    categories = sorted(label_to_id.keys())
    return file_paths, labels, categories


class MusicDataset(Dataset):
    def __init__(self, mode, x, y, categories, mixup, filtaug):
        self.data = []
        self.labels = y
        self.mode = mode
        self.categories = categories
        self.mean = 0
        self.std = 0
        for ind in tqdm(range(len(x)), desc=self.mode + ' dataset', ncols=37):
            file_path = x[ind]
            file_path = os.path.join(MUSIC_DATASET_PATH, file_path)
            audio = AudioUtil.open(file_path)
            audio = AudioUtil.resample(audio, SAMPLE_RATE)
            audio = AudioUtil.rechannel(audio, 1)
            audio = AudioUtil.pad_trunc(audio, 4000)
            sgram = AudioUtil.get_spectrogram(audio, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            if filtaug:
                # sgram_squeezed = sgram.squeeze()
                sgram = filt_aug(sgram)
                # sgram = sgram.unsqueeze(1)
            if mixup:
                mixer = MixupBYOLA(ratio=0.2, log_mixup_exp=True)
                sgram = mixer(sgram)
            self.data.append(sgram)
            self.mean += sgram.mean()
            self.std += sgram.std()

        self.mean /= len(x)
        self.std /= len(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # normalize spectrogram using dataset mean and stddev
        sgram = self.data[idx]
        sgram = (sgram - self.mean) / self.std  # weg
        return sgram, self.labels[idx]
