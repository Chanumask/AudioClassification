import json
import pickle

import pandas as pd
import seaborn as sns

import librosa.display
import matplotlib.pyplot as plt
from matplotlib import pylab

from AudioUtil import *
from settings import *


def save_object(obj):
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def plot_example(title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    audio_path = "data/ESC-50-master/audio/"
    path = audio_path + '1-100038-A-14.wav'
    audio = AudioUtil.open(path)
    sgram = AudioUtil.get_spectrogram(audio, n_mels=NMELS, n_fft=NFFT, hop_len=HOPLENGTH)
    sgram = torch.squeeze(sgram, dim=0)
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(sgram, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    return plt


# PLOT FROM WAVEFORM (LIBROSA)
def plot_wave(signal):
    # signal = signal.numpy()
    librosa.display.waveshow(y=signal, sr=SAMPLE_RATE, alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    return plt


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()
    x, num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    return plt


def plot_power_spectrum(signal):
    fft = np.fft.fft(signal)  # Fourier transform
    spectrum = np.abs(fft)
    f = np.linspace(0, SAMPLE_RATE, len(spectrum))
    left_spectrum = spectrum[:int(len(spectrum) / 2)]
    left_f = f[:int(len(spectrum) / 2)]

    plt.plot(left_f, left_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")
    return plt


def plot_spec(signal):
    # signal = signal.numpy()
    stft = librosa.stft(signal, n_fft=NFFT, hop_length=HOPLENGTH)
    spectrogram = np.abs(stft)

    librosa.display.specshow(spectrogram, sr=SAMPLE_RATE, hop_length=HOPLENGTH, cmap='viridis')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Spectrogram")
    return plt


def plot_db_spectrogram(signal):
    hop_length = 512
    n_fft = 2048
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.amplitude_to_db(np.abs(stft))

    librosa.display.specshow(log_spectrogram, sr=SAMPLE_RATE, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    return plt


def plot_mfccs(signal):
    hop_length = 512
    n_fft = 2048
    # extract 13 Mel Frequency Cepstral Coefficients
    MFCCs = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    librosa.display.specshow(data=MFCCs, sr=SAMPLE_RATE, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    return plt


def plot_mel_spectrogram(signal):
    hop_length = 512

    S = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=128)
    S_db_mel = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    librosa.display.specshow(S_db_mel, sr=SAMPLE_RATE, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (dB)")
    return plt


def get_dataset_path(dataset_name):
    if dataset_name == "SPEECH":
        return SPEECH_DATASET_PATH
    elif dataset_name == "MUSIC":
        return MUSIC_DATASET_PATH
    elif dataset_name == "PRIMATES":
        return PRIMATES_DATASET_PATH
    elif dataset_name == "ESC50":
        return ESC50_DATASET_PATH
    else:
        raise ValueError("Invalid dataset name.")


def plot_all(dataset_name):
    dataset_path = get_dataset_path(dataset_name)

    if os.path.isdir(dataset_path):
        # Dataset path points to a directory -> pick a random sample
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if len(subdirs) > 0:
            # Dataset path points to a directory of subdirectories containing samples
            subdir_path = os.path.join(dataset_path, random.choice(subdirs))
            sample_file = random.choice(os.listdir(subdir_path))
            sample_path = os.path.join(subdir_path, sample_file)
        else:
            # Dataset path points to a directory of samples
            sample_file = random.choice(os.listdir(dataset_path))
            sample_path = os.path.join(dataset_path, sample_file)
    else:
        # Dataset path points to a single file
        sample_path = dataset_path

    signal, sr = librosa.load(sample_path, sr=None, mono=True)
    plt.figure(figsize=(15, 10))
    plt.figure.__name__ = "test"
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.subplot(3, 2, 1)
    plot_wave(signal)
    plt.subplot(3, 2, 2)
    plot_power_spectrum(signal)
    plt.subplot(3, 2, 3)
    plot_spec(signal)
    plt.subplot(3, 2, 4)
    plot_db_spectrogram(signal)
    plt.subplot(3, 2, 5)
    plot_mfccs(signal)
    plt.subplot(3, 2, 6)
    plot_mel_spectrogram(signal)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title(f"class is: {sr}")
    plt.show()
    return sample_path


# PLOT FROM WAVE FORM (TORCHAUDIO)
def plot_spectrogram(sgram):
    plt.figure(figsize=(10, 4))
    plt.imshow(sgram.squeeze().numpy(), origin="lower", cmap="inferno")
    plt.xlabel("Time")
    plt.ylabel("Mel frequency")
    plt.colorbar().set_label("dB")
    plt.tight_layout()
    plt.show()


# PLOT RESULTS
def liveplot(result_list, pairs):
    print(pairs)
    plt.close()
    plt.ion()
    plt.show()
    cols = ['crimson', 'tab:blue', 'peru', 'darkolivegreen', 'k']
    fig, ax = plt.subplots(len(pairs), figsize=(10, 10))
    plt.subplots_adjust(left=0.08,
                        bottom=0.05,
                        right=0.98,
                        top=0.95,
                        wspace=0.4,
                        hspace=0.52)
    for i, pair in enumerate(pairs):
        for title, graphs in pair.items():
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(title)

            for graph in graphs:
                ax[i].plot([res[graph] for res in result_list], '-x', color=cols[i])
            plt.draw()
            plt.pause(0.0001)


def find_max_acc_epoch(results):
    max_acc = 0.0
    max_acc_epoch = 0
    for i, res in enumerate(results):
        if res['avg_valid_acc'] > max_acc:
            max_acc = res['avg_valid_acc']
            max_acc_epoch = i  # adding 1 to the axis later instead of this value, so that it starts at epoch 1 not 0
    return max_acc, max_acc_epoch


def plot_results(result_list):
    colors = sns.color_palette("bright", 10)
    fig, ax = plt.subplots(5, figsize=(12, 10))
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.95, wspace=0.4, hspace=0.52)

    label_distance = 50

    for i in range(len(ax)):
        ax[i].set_xticks(range(0, EPOCHS, label_distance))
        ax[i].set_xticklabels(range(0, EPOCHS, label_distance))
        ax[i].set_xlabel("Epochs")

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("F1 Score")
    ax[2].set_ylabel("Losses")
    ax[3].set_ylabel("UAR")
    ax[4].set_ylabel("Learning Rate")

    max_acc_value, max_acc_epoch = find_max_acc_epoch(result_list)
    max_acc_value = round(max_acc_value * 100, 2)
    ax[0].plot([res['avg_valid_acc'] for res in result_list], marker='o', color=colors[0])
    ax[0].set_ylim([0, 1])
    ax[0].axvline(x=max_acc_epoch, linestyle='dotted', color='black')
    ax[0].text(x=(max_acc_epoch + 0.1), y=0.2, s=f'Max Acc: {max_acc_value}', rotation=90, size='x-small')
    ax[1].plot([res['f1'] for res in result_list], marker='o', color=colors[1])
    ax[2].plot([res['avg_train_loss'] for res in result_list], marker='o', color=colors[3], label='Train Loss')
    ax[2].plot([res['avg_valid_loss'] for res in result_list], marker='^', color=colors[6], label='Valid Loss')
    ax[2].legend()
    ax[3].plot([res['uar'] for res in result_list], marker='o', color='k')
    ax[4].plot([res['lr'] for res in result_list], marker='o', color='k')

    plt.savefig('results/plotted_training_results.png', dpi=300, bbox_inches="tight")


def tabulate_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(
        columns=["comment", "Max acc", "Max UAR", "seed", "dual patchnorm", "random resize crop", "random linear fade",
                 "mixup", "spec_aug",
                 "kernelsize",
                 "stride", "AdamW wd"])

    for i in range(0, len(data), 4):
        params, acc, uar, seed = data[i], data[i + 1], data[i + 2], data[i + 3]
        lr, rcc, rlf, dual_patchnorm, mixup, init_kernel_size, init_stride, weight_decay, ema, spec_aug, mask_prob, comment = params

        df.loc[i // 2] = [comment, acc, uar, seed, dual_patchnorm, rcc, rlf, mixup, spec_aug, init_kernel_size,
                          init_stride,
                          weight_decay]
    df = df.sort_values(by=['Max acc'], ascending=False)
    pd.set_option('display.max_columns', None)

    print(df)


def average10seeds():
    rootdir = "results/10seeds"
    results = []
    for subdir, dirs, files in os.walk(rootdir):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(subdir, file_name)
                # print(file_path)

                with open(file_path) as f:
                    data = json.load(f)

                accuracies = []
                uars = []

                for i in range(0, len(data), 4):
                    accuracy = data[i + 1]
                    uar = data[i + 2]

                    accuracies.append(accuracy)
                    uars.append(uar)

                avg_accuracy = sum(accuracies) / len(accuracies)
                avg_accuracy = round(100 * avg_accuracy, 2)
                avg_uar = sum(uars) / len(uars)
                avg_uar = round(100 * avg_uar, 2)

                if file_name.endswith("PRIMATES.json"):
                    minimum = min(uars)
                    maximum = max(uars)
                else:
                    minimum = min(accuracies)
                    maximum = max(accuracies)

                setting = os.path.basename(subdir)

                results.append({
                    'Setting': setting,
                    'File Name': file_name.split(".")[0],
                    'Avg. Accuracy': avg_accuracy,
                    'Avg. UAR': avg_uar,
                    'Minimum': minimum,
                    'Maximum': maximum,
                })

    df = pd.DataFrame(results)

    def get_baseline_acc(file_name):
        baseline_df = df[(df['Setting'] == 'no data augmentations') & (df['File Name'] == file_name)]
        if len(baseline_df) > 0:
            return baseline_df.iloc[0]['Avg. Accuracy']
        else:
            return None

    df['Avg. improvement'] = df['File Name'].apply(get_baseline_acc)
    df['Avg. improvement'] = df['Avg. Accuracy'] - df['Avg. improvement']
    # print(df[(df['File Name'] == 'MUSIC.json')])
    return df


def bar_plot_averages(df, settings):
    results = {}
    for setting in settings:
        results[setting] = {'acc': [], 'min': [], 'max': []}  # initialize empty dictionary for each setting
        for file_name in df['File Name'].unique():
            if file_name == "PRIMATES":
                avg = df[(df['Setting'] == setting) & (df['File Name'] == file_name)]['Avg. UAR'].values[
                    0]
            else:
                avg = df[(df['Setting'] == setting) & (df['File Name'] == file_name)]['Avg. Accuracy'].values[
                    0]
            results[setting]['acc'].append(avg)
            minimum = avg - round(df[(df['Setting'] == setting) & (df['File Name'] == file_name)]['Minimum'].values[
                                      0] * 100, 1)
            maximum = round(df[(df['Setting'] == setting) & (df['File Name'] == file_name)]['Maximum'].values[
                                0] * 100, 1) - avg
            results[setting]['min'].append(minimum)
            results[setting]['max'].append(maximum)
            # print(results[setting])

    num_datasets = len(df['File Name'].unique())
    bar_width = 0.8 / len(settings)

    plt.figure(figsize=(10, 7))
    xpos = np.arange(num_datasets)

    for i, setting in enumerate(settings):
        yerr = [results[setting]['min'], results[setting]['max']]
        # print(results[setting]['acc'])
        plt.bar(xpos + i * bar_width, results[setting]['acc'], color=sns.color_palette("colorblind")[i], width=bar_width,
                label=setting.title(), yerr=yerr, capsize=5)
        for w, (x, y) in enumerate(zip(xpos + i * bar_width, results[setting]['acc'])):
            plt.text(x, y - results[setting]['min'][w] - 3, str(round(y, 2)), ha='center', va='bottom', fontsize=7)

    plt.xticks(xpos + bar_width * len(settings) / 2 - 0.08, df['File Name'].unique())
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy/ UAR (%)')
    plt.legend(loc='lower right')
    plt.show()
