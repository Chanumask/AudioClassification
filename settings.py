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
    'comment': ["300 Epochs"]
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
    'comment': ["300 Epochs"]
}

# General
DATASET = "MUSIC"  # "SPEECH", "ESC50", "MUSIC"
RNG_SEED = 1  # 2, 1, 1

ONLY_TABULATE = False
SAVE_DATA = True
SAVE_MAX_MODEL = True
PLOT_RES = True
MONITORING = False
UPDATE_INTERVAL = 5  # every x Epochs
CONF_MATRIX = True

# Training Loop
BATCH_SIZE = 16  # 32, 64
EPOCHS = 300
LR_WARUMUP = True
COSINE = True
EMA_ON = False
EMA_START = 0.7

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_JSON_PATH = "data/SpeechCommands/speechData.json"
MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_JSON_PATH = "data/GTZAN/musicData.json"

# Spectrogram
SAMPLE_RATE = 22050
NFFT = 1024
HOPLENGTH = 512
NMELS = 128
FMIN = 20
FMAX = 8300
TOPDB = 80
