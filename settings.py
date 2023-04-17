HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'rrc': [False],
    'rlf': [False],
    'dual_patchnorm': [False],
    'mixup': [False],  # only 1 value possible atm
    'init_kernel_size': [3],  # [(8, 4), (16, 8), (16, 4)],
    'init_stride': [2],  # [(4, 2), (8, 4)],
    'weight_decay': [1e-3],  # 1e-2
    'seed': [2],
    'ema': [False],
    'spec_aug': [False],
    'comment': [""]
}  # add new hyperparams above the comment entry

HYPERPARAMS_ESC50 = {
    'learning_rate': [2e-4],
    'rrc': [False],
    'rlf': [False],
    'dual_patchnorm': [False],
    'mixup': [False],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-3],
    'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [False],
    'spec_aug': [False],
    'mask_prob': [0],
    'comment': ["adamw"]
}

HYPERPARAMS_MUSIC = {
    'learning_rate': [1e-4],
    'rrc': [False],
    'rlf': [False],
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-2],
    'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [False],
    'spec_aug': [False],
    'mask_prob': [0],
    'comment': ["mixup"],
}

HYPERPARAMS_PRIMATES = {
    'learning_rate': [1e-4],
    'rrc': [False],
    'rlf': [False],
    'dual_patchnorm': [False],
    'mixup': [False],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-2],
    'seed': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [False],
    'spec_aug': [False],
    'mask_prob': [0],
    'comment': ["adamw"]
}

# General
DATASET = "MUSIC"  # "SPEECH", "ESC50", "MUSIC", "PRIMATES"
ONLY_TABULATE = False
ONLY_PLOT_EXAMPLE = False
MAJORITY_VOTE = False
AVG_SEEDS = False
BARPLOT_SETTING = ["baseline", "rlf", "rrc", "proper_cosine"]

USE_MAX_MODEL = False
PLOT_RES = True
MONITORING = False
UPDATE_INTERVAL = 1  # every x Epochs
CONF_MATRIX = False

SAVE_DATA = True
SAVE_PATH = "10seeds//adamw"
SAVE_MAX_MODEL = True
SAVE_ENSEMBLE = False
ENSEMBLE_NAME = "final_results"

# Training Loop
BATCH_SIZE = 32  # 32, 64
EPOCHS = 80
LR_WARUMUP = True
COSINE = False

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_PATH_TRAIN = "data/SpeechCommands/training_list.txt"
SPEECH_PATH_VALID = "data/SpeechCommands/validation_list.txt"

MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_PATH_TRAIN = "data/GTZAN/training_list.txt"
MUSIC_PATH_VALID = "data/GTZAN/validation_list.txt"

PRIMATES_DATASET_PATH = "data/Primates/wav"
PRIMATES_CSV_TRAIN = "data/Primates/lab/train.csv"
PRIMATES_CSV_VALID = "data/Primates/lab/devel.csv"

ESC50_DATASET_PATH = "data/ESC-50-master/audio"

# Spectrogram
SAMPLE_RATE = 22050
NFFT = 1024
HOPLENGTH = 512
NMELS = 128
FMIN = 20
FMAX = 8300
TOPDB = 80
