from torch.utils.data import Dataset

from AudioUtil import *
from settings import *


def load_primates_dataset(csv_path):
    df = pd.read_csv(csv_path)
    x = np.array(df["filename"])
    y = np.array(df["label"])
    categories = sorted(df['label'].unique())
    return x, y, categories


class PrimatesDataset(Dataset):
    def __init__(self, mode, x, y, categories, specaug, mask_prob):
        self.data = []
        self.labels = y
        self.c2i = {}
        self.i2c = {}
        self.mode = mode
        self.categories = categories

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
            sgram = AudioUtil.get_spectrogram(audio, specaug=specaug, mask_prob=mask_prob, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            self.data.append(sgram)
            self.labels[ind] = (self.c2i[self.labels[ind]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
