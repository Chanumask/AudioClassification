import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from termcolor import colored
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.models import resnet34, ResNet34_Weights
import os
import seaborn as sns
from itertools import cycle
import logging
from colorlog import ColoredFormatter

import AudioUtil
import utilsPlotting
from ESC50Dataset import ESC50Data
from settings import *


# SETTINGS
def settings():
    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# LOGGING
def initiateLogging():
    LOG_LEVEL = logging.DEBUG
    LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(formatter)
    log = logging.getLogger('pythonConfig')
    log.setLevel(LOG_LEVEL)
    log.addHandler(stream)

    # Demo
    # log.debug("A quirky message only developers care about")
    # log.info("Curious users might want to know this")
    # log.error("Serious stuff, this is red for a reason")
    # log.critical("OH NO everything is on fire")
    return log


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def get_melspectrogram_db(file_path, sr=SAMPLE_RATE, n_fft=NFFT, hop_length=HOPLENGTH, n_mels=NMELS, fmin=FMIN,
                          fmax=FMAX, top_db=TOPDB):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def lr_decay(optimizer, epoch):
    if epoch % 20 == 0:
        new_lr = LEARNING_RATE / (10 ** (epoch // 20))
        optimizer = setlr(optimizer, new_lr)
        log.info(f'Changed learning rate to {new_lr}')
    return optimizer


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    log.info("Starting training...")
    epochId = 0
    resultarray = []
    for epoch in (range(1, epochs + 1)):
        epochId += 1
        log.info(f"Epoch {epochId} / {epochs}")
        model.train()
        batch_losses = []
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in tqdm(enumerate(train_loader), desc='Epoch progress', ncols=33):
            x, y = data
            # print(x.shape) # (64, 1, 128, 431)
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)  # y.float() for mixup?
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        # print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        batch_losses, trace_y, trace_yhat, lrs = [], [], [], []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        valid_loss = np.mean(valid_losses[-1])
        f1_score = 1
        resultarray.append(
            {'avg_valid_loss': valid_loss, 'avg_valid_acc': accuracy, 'avg_train_loss': train_losses,
             'lr': lrs, 'f1': f1_score})
        # print(f'Epoch - {epoch} Valid-Loss : {valid_loss} Valid-Accuracy : {accuracy}')
    return resultarray


def plot_example(path, valid_dataframe):
    wav, sr = librosa.load(path + '1-100032-A-0.wav', sr=None)
    print(f'Sampling rate of the audio is {sr} and length of the audio is {len(wav) / sr} seconds')

    filename = valid_dataframe[valid_dataframe['category'] == 'clock_tick'].iloc[0]['filename']
    wav, sr = librosa.load(path + filename, sr=None)
    # librosa.display.waveshow(y=wav, sr=sr)
    # plt.show()
    spec = get_melspectrogram_db(audio_path + filename, sr)
    librosa.display.specshow(spec_to_image(spec), cmap='viridis')
    plt.show()
    # spec = utilsAugmenting.spec_augment(spec)
    librosa.display.specshow(spec_to_image(spec), cmap='viridis')
    plt.show()


if __name__ == "__main__":

    settings()
    log = initiateLogging()

    log.info('Loading and preprocessing datasets...')
    df = pd.read_csv("data/ESC-50-master/meta/esc50.csv")
    audio_path = "data/ESC-50-master/audio/"

    train_df = df[df['fold'] != 5]
    valid_df = df[df['fold'] == 5]

    # plot_example(audio_path, valid_df)

    train_data = ESC50Data("training", audio_path, train_df, 'filename', 'category')
    valid_data = ESC50Data("validation", audio_path, valid_df, 'filename', 'category')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"selected device: {device}")

    resnet_model = resnet34(weights=ResNet34_Weights.DEFAULT)
    resnet_model.fc = nn.Linear(512, 50)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)

    lr = LEARNING_RATE
    epochs = EPOCHS
    optimizer = optim.Adam(resnet_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    resnet_train_losses = []
    resnet_valid_losses = []

    results = train(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses,
                    resnet_valid_losses, lr_decay)

    # print(f"final results: {results}")

    # tl = np.asarray(resnet_train_losses).ravel()
    # vl = np.asarray(resnet_valid_losses).ravel()
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(tl)
    # plt.legend(['Train Loss'])
    # plt.subplot(1, 2, 2)
    # plt.plot(vl, 'orange')
    # plt.legend(['Valid Loss'])
    # plt.show()

    log.info("\ntraining accuracies in steps:")

    for result in results:
        print(colored(result['avg_valid_acc'], 'green'))

    log.info("Plotting results")
    utilsPlotting.plot_results(results, [{"Accuracies vs epochs": ['avg_valid_acc']}, {"f1_score vs epochs": ['f1']},
                                         {"Losses vs epochs": ['avg_valid_loss']},
                                         {"Learning rates per batch vs epochs": ['lr']}], epoch=1)

    with open('esc50resnet.pth', 'wb') as f:
        torch.save(resnet_model, f)
