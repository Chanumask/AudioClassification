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


def preprocess_music_dataset(dataset_path, json_path):
    # dictionary where we'll store mapping, labels and filenames
    data = {
        "mapping": [],
        "labels": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for j, f in enumerate(filenames):
                file_path = os.path.join(dirpath, f)

                # store data for analysed track
                data["labels"].append(i - 1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i - 1))

        # save data in json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["files"])
    y = np.array(data["labels"])
    m = np.array(data["mapping"])
    return X, y, m


def split_music_dataset(data_path, test_size=0.2, validation_size=0.2):
    X, y, m = load_data(data_path)
    categories = []
    for i in m:
        i = i.split("\\", 1)[1]
        categories.append(i)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, y_train, X_validation, y_validation, X_test, y_test, categories


class MusicDataset(Dataset):
    def __init__(self, mode, X, y, categories, mixup):
        self.data = []
        self.labels = y
        self.c2i = {}
        self.i2c = {}
        self.mode = mode
        self.categories = categories
        for ind in tqdm(range(len(X)), desc=self.mode + ' dataset', ncols=37):
            file_path = X[ind]
            audio = AudioUtil.open(file_path)
            audio = AudioUtil.resample(audio, SAMPLE_RATE)
            audio = AudioUtil.rechannel(audio, 1)
            audio = AudioUtil.pad_trunc(audio, 4000)
            sgram = AudioUtil.get_spectrogram(audio, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            if mixup:
                mixer = MixupBYOLA(ratio=0.2, log_mixup_exp=True)
                sgram = mixer(sgram)
            self.data.append(sgram)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx].shape)  # (1, 128, 431)
        return self.data[idx], self.labels[idx]
