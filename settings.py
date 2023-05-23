# Model training
DATASET = "PRIMATES"  # "SPEECH", "ESC50", "MUSIC", "PRIMATES"
SAVE_DATA = True
SAVE_PATH = "10seeds//new experiment"
SAVE_MAX_MODEL = True
USE_MAX_MODEL = False
SAVE_ENSEMBLE = False
ENSEMBLE_NAME = "best configuration 300"

# Displaying and plotting results
PLOT_RES = True
CONF_MATRIX = False
MONITORING = False
UPDATE_INTERVAL = 10  # every x Epochs
AVG_SEEDS = False
BARPLOT_SETTING = ["baseline", "random resize crop", "random linear fade", "mixup", "augmentations combined"]
# BARPLOT_SETTING = ["baseline", "cosine annealing", "optimized adamW", "best augmentations", "augmentations and training loop"]
# BARPLOT_SETTING = ["augmentations and training loop", "no sequential pooling", "4 attention heads"]
# BARPLOT_SETTING = ["augmentations and training loop", "dual patchnorm", "batch normalization", "improve locality",
#                    "l2 norm + trainable scale", "32 memory vectors", "grn mlp", "best configuration"]
# BARPLOT_SETTING = ["baseline", "baseline with EMA", "best configuration", "best configuration, ema"]
# BARPLOT_SETTING = ["best configuration", "best configuration 300"]

MAJORITY_VOTE = False
ONLY_TABULATE = False
ONLY_PLOT_EXAMPLE = False

# Training Loop
BATCH_SIZE = 16  # 32, 64
EPOCHS = 15
LR_WARUMUP = True
COSINE = True

# Model parameters and extensions
HYPERPARAMS_PRIMATES = [
    {
        'learning_rate': [1e-4],
        'random resize crop': [True],
        'random linear fade': [False],
        'dual patchnorm': [True],
        'mixup': [False],  # only 1 value possible atm
        'init_kernel_size': [3],
        'init_stride': [2],
        'weight_decay': [1e-2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'ema': [False],
        'spec_aug': [False],
        'mask_prob': [0],
        'norm': ["ln"],
        'num mem vecs': [0],
        'improve locality': [True],
        'scale value': [1.0],
        'trainable scale': [False],
        'l2': [False],
        'use grn mlp': [True],
        'seq pool': [True],
        'num heads': [8],
        'comment': ["added softmax"]
    }
]

HYPERPARAMS_ESC50 = [
    {
        'learning_rate': [2e-4],
        'random resize crop': [True],
        'random linear fade': [False],
        'dual patchnorm': [False],
        'mixup': [False],  # only 1 value possible atm
        'init_kernel_size': [3],
        'init_stride': [2],
        'weight_decay': [1e-3],
        'seed': [9],
        'ema': [False],
        'spec_aug': [False],
        'mask_prob': [0],
        'norm': ["ln"],
        'num mem vecs': [0],
        'improve locality': [False],
        'scale value': [1.0],
        'trainable scale': [False],
        'l2': [False],
        'use grn mlp': [True],
        'seq pool': [True],
        'num heads': [8],
        'comment': ["best configuration, no ema 300"]
    }
]

HYPERPARAMS_MUSIC = [
    {
        'learning_rate': [2e-4],
        'random resize crop': [True],
        'random linear fade': [False],
        'dual patchnorm': [True],
        'mixup': [False],  # only 1 value possible atm
        'init_kernel_size': [3],
        'init_stride': [2],
        'weight_decay': [1e-2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'ema': [False],
        'spec_aug': [False],
        'mask_prob': [0],
        'norm': ["ln"],
        'num mem vecs': [32],
        'improve locality': [True],
        'scale value': [1.0],
        'trainable scale': [False],
        'l2': [False],
        'use grn mlp': [True],
        'seq pool': [False],
        'num heads': [4],
        'comment': ["best configuration, no ema 300"],
    }
]

HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'random resize crop': [False],
    'random linear fade': [False],
    'dual patchnorm': [False],
    'mixup': [False],  # only 1 value possible atm
    'init_kernel_size': [3],  # [(8, 4), (16, 8), (16, 4)],
    'init_stride': [2],  # [(4, 2), (8, 4)],
    'weight_decay': [1e-3],  # 1e-2
    'seed': [2],
    'ema': [False],
    'spec_aug': [False],
    'comment': [""]
}

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_PATH_TRAIN = "data/SpeechCommands/training_list.txt"
SPEECH_PATH_VALID = "data/SpeechCommands/validation_list.txt"

MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_PATH_TRAIN = "data/GTZAN/training_list.txt"
MUSIC_PATH_VALID = "data/GTZAN/validation_list.txt"
MUSIC_PATH_TEST = "data/GTZAN/testing_list.txt"

PRIMATES_DATASET_PATH = "data/Primates/wav"
PRIMATES_CSV_TRAIN = "data/Primates/lab/train.csv"
PRIMATES_CSV_VALID = "data/Primates/lab/devel.csv"
PRIMATES_CSV_TEST = "data/Primates/lab/test.csv"

ESC50_DATASET_PATH = "data/ESC-50-master/audio"

# Spectrogram parameters
SAMPLE_RATE = 22050
NFFT = 1024
HOPLENGTH = 512
NMELS = 128
FMIN = 20
FMAX = 8300
TOPDB = 80
