import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from AudioUtil import *
from settings import *


def load_primate_data(csv_path):
    df = pd.read_csv(csv_path)

    x = np.array(df["filename"])
    y = np.array(df["label"])
    categories = sorted(df['label'].unique())
    return x, y, categories


def split_primates_dataset(csv_path, test_size=0.2, validation_size=0.2):
    x, y, categories = load_primate_data(csv_path)

    # create splits
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, y_train, x_validation, y_validation, x_test, y_test, categories


class PrimatesDataset(Dataset):
    def __init__(self, mode, x, y, categories, mixup):
        self.data = []
        self.labels = y
        self.c2i = {}
        self.i2c = {}
        self.mode = mode
        self.categories = categories
        self.mean = 0
        self.std = 0

        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category

        for ind in tqdm(range(len(x)), desc=self.mode + ' dataset', ncols=37):
            file_path = x[ind]
            file_path = PRIMATES_DATASET_PATH + "/" + file_path
            audio = AudioUtil.open(file_path)
            audio = AudioUtil.resample(audio, SAMPLE_RATE)
            audio = AudioUtil.rechannel(audio, 1)
            audio = AudioUtil.pad_trunc(audio, 4000)
            sgram = AudioUtil.get_spectrogram(audio, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            if mixup:
                mixer = MixupBYOLA(ratio=0.2, log_mixup_exp=True)
                sgram = mixer(sgram)
            self.data.append(sgram)
            self.mean += sgram.mean()
            self.std += sgram.std()
            self.labels[ind] = (self.c2i[self.labels[ind]])

        self.mean /= len(x)
        self.std /= len(x)
        print('Mean:', self.mean, 'Std:', self.std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # normalize spectrogram using dataset mean and stddev
        sgram = self.data[idx]
        sgram = (sgram - self.mean) / self.std
        return sgram, self.labels[idx]
