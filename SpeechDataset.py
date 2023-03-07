import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from AudioUtil import AudioUtil, MixupBYOLA
from settings import *
from tqdm import tqdm

SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio


def preprocess_speech_dataset(dataset_path, json_path):
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
                if j == 50:
                    break
                file_path = os.path.join(dirpath, f)

                # store data for analysed track
                data["labels"].append(i - 1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i - 1))

        # save data in json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["files"])
    y = np.array(data["labels"])
    m = np.array(data["mapping"])
    return X, y, m


def split_speech_dataset(data_path, test_size=0.2, validation_size=0.2):
    X, y, m = load_data(data_path)
    categories = []
    for i in m:
        i = i.split("\\", 1)[1]
        categories.append(i)
    # create train, validation, test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, y_train, x_validation, y_validation, x_test, y_test, categories


class SpeechDataset(Dataset):
    def __init__(self, mode, x, y, categories, mixup):
        self.data = []
        self.labels = y
        self.c2i = {}
        self.i2c = {}
        self.mode = mode
        self.categories = categories
        for ind in tqdm(range(len(x)), desc=self.mode + ' dataset', ncols=37):
            file_path = x[ind]
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
