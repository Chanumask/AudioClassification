HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(8, 4)],
    'init_stride': [(4, 2)],
    'weight_decay': [1e-2],
    'seed': [0, 4, 5, 8],
    'ema': [True],
    'filt_aug': [False],
    'comment': ["filt_aug after rrc"]
}  # add new hyperparams above the comment entry

HYPERPARAMS_ESC50 = {
    'learning_rate': [2e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-3],
    'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [True],
    'filt_aug': [False],
    'comment': ["ema start=1"]
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
    'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [True],
    'filt_aug': [False],
    'comment': ["ema start=1"],
}

HYPERPARAMS_PRIMATES = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2],
    'weight_decay': [1e-3],  # try 2 or 4
    'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'ema': [True],
    'filt_aug': [False],
    'comment': [""]
}

# General
DATASET = "SPEECH"  # "SPEECH", "ESC50", "MUSIC", "PRIMATES"
ONLY_TABULATE = False
ONLY_PLOT_EXAMPLE = False
MAJORITY_VOTE = False

USE_MAX_MODEL = False
PLOT_RES = False
MONITORING = False
UPDATE_INTERVAL = 1  # every x Epochs
CONF_MATRIX = False

SAVE_DATA = True
SAVE_MAX_MODEL = False
SAVE_ENSEMBLE = True
ENSEMBLE_NAME = "300"

# Training Loop
BATCH_SIZE = 16  # 32, 64
EPOCHS = 300
LR_WARUMUP = True
COSINE = True

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
