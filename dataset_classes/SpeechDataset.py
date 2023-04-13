from torch.utils.data import Dataset

import os
import numpy as np


from AudioUtil import AudioUtil
from settings import *
from tqdm import tqdm

SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio


def preprocess_speech_dataset(dataset_path, json_path):
    validation_file = "data/SpeechCommands/validation_list.txt"
    testing_file = "data/SpeechCommands/testing_list.txt"

    # Iterate through each category folder and check if its files are in validation or testing set
    training_files = []
    for category in os.listdir("data/SpeechCommands"):
        if category.startswith("."):  # ignore hidden files/folders
            continue
        category_path = os.path.join("data/SpeechCommands", category)
        if not os.path.isdir(category_path):  # skip files that are not directories
            continue
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                file_path = os.path.join(category, file)
                if file_path not in open(validation_file).read() and file_path not in open(testing_file).read():
                    training_files.append(file_path)

    # Write the list of training files to a new txt file
    with open("training_list.txt", "w") as f:
        for file_path in training_files:
            f.write(file_path + "\n")


def remove_duplicates(training_file_path, valid_file_path, test_file_path):
    with open(valid_file_path) as f:
        valid_files = set(line.strip() for line in f)
    with open(test_file_path) as f:
        test_files = set(line.strip() for line in f)

    with open(training_file_path, "r") as f:
        training_files = f.read().splitlines()

    training_files = [f.replace("\\", "/") for f in training_files]
    new_training_files = list(filter(lambda x: x not in valid_files and x not in test_files, training_files))

    with open(training_file_path, "w") as f:
        f.write("\n".join(new_training_files))


def count_dataset_files(train_path, valid_path, test_path, dataset_path):
    train_files = set(line.strip() for line in open(train_path))
    valid_files = set(line.strip() for line in open(valid_path))
    test_files = set(line.strip() for line in open(test_path))
    total_files = 0
    for category in os.listdir(dataset_path):
        if category.startswith("."):  # ignore hidden files/folders
            continue
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):  # skip files that are not directories
            continue
        total_files += len([name for name in os.listdir(category_path) if name.endswith(".wav")])
    return total_files, len(train_files), len(valid_files), len(test_files)


def create_label_to_id_dict(dataset_path):
    """Creates a dictionary that maps each label to a unique integer ID."""
    labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]
    labels.sort()  # sort labels to ensure consistent ordering
    label_to_id = {label: i for i, label in enumerate(labels)}
    return label_to_id


def load_speech_dataset(data_file_path):
    label_to_id = create_label_to_id_dict(SPEECH_DATASET_PATH)
    with open(data_file_path, "r") as f:
        file_paths = f.read().splitlines()

    labels = np.zeros(len(file_paths), dtype=int)

    for i, path in enumerate(file_paths):
        label = path.split("/")[0]
        label_id = int(label_to_id[label])
        labels[i] = label_id

    categories = sorted(label_to_id.keys())
    return file_paths, labels, categories


class SpeechDataset(Dataset):
    def __init__(self, mode, x, y, categories, specaug, mask_prob):
        self.data = []
        self.labels = y
        self.c2i = {}
        self.i2c = {}
        self.mode = mode
        self.categories = categories

        for ind in tqdm(range(len(x)), desc=self.mode + ' dataset', ncols=37):
            file_path = x[ind]
            file_path = os.path.join(SPEECH_DATASET_PATH, file_path)
            audio = AudioUtil.open(file_path)
            audio = AudioUtil.resample(audio, SAMPLE_RATE)
            audio = AudioUtil.rechannel(audio, 1)
            audio = AudioUtil.pad_trunc(audio, 4000)
            sgram = AudioUtil.get_spectrogram(audio, specaug=specaug, mask_prob=mask_prob, n_mels=NMELS, n_fft=NFFT, hop_len=None)
            self.data.append(sgram)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
