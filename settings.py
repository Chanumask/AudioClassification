HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(8, 4)],
    'init_stride': [(4, 2)],
    'weight_decay': [1e-2],
    'comment': [""]
}

HYPERPARAMS_ESC50 = {
    'learning_rate': [2e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-3],
    'comment': ["EMA 0.7"]
}

HYPERPARAMS_MUSIC = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [True],  # RandomLinearFade
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(3, 3)],
    'init_stride': [(2, 2)],
    'weight_decay': [1e-3],
    'comment': ["EMA 0.7"]
}

HYPERPARAMS_PRIMATES = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-3],     # try 2 or 4
    'comment': ["EMA = False"]
}

# General
DATASET = "PRIMATES"  # "SPEECH", "ESC50", "MUSIC", "PRIMATES"
RNG_SEED = 2  # 2, 1, 1, 2
ONLY_TABULATE = False
ONLY_PLOT_EXAMPLE = False
SAVE_DATA = True

SAVE_MAX_MODEL = True
USE_MAX_MODEL = False
PLOT_RES = True
MONITORING = False
UPDATE_INTERVAL = 1  # every x Epochs
CONF_MATRIX = False

# Training Loop
BATCH_SIZE = 16  # 32, 64
EPOCHS = 120
LR_WARUMUP = True
COSINE = True
EMA_ON = False
EMA_START = 0.7

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_JSON_PATH = "data/SpeechCommands/speechData.json"
MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_JSON_PATH = "data/GTZAN/musicData.json"
PRIMATES_DATASET_PATH = "data/Primates/wav"
PRIMATES_CSV = "data/Primates/lab/train.csv"
ESC50_DATASET_PATH = "data/ESC-50-master/audio"

# Spectrogram
SAMPLE_RATE = 22050
NFFT = 1024
HOPLENGTH = 512
NMELS = 128
FMIN = 20
FMAX = 8300
TOPDB = 80
