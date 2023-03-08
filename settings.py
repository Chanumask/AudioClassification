HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(8, 4)],
    'init_stride': [(4, 2)],
    'weight_decay': [1e-1, 1e-2, 1e-3]
}

HYPERPARAMS_ESC50 = {
    'learning_rate': [1e-4, 2e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2]
}

HYPERPARAMS_MUSIC = {
    'learning_rate': [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
    'rrc': [True],  # RandomResizeCrop
    'rlf': [True],  # RandomLinearFade
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(3, 3)],
    'init_stride': [(2, 2)]
}

# General
DATASET = "SPEECH"  # "SPEECH", "ESC50", "MUSIC"
RNG_SEED = 2

ONLY_TABULATE = False
SAVE_DATA = True
SAVE_MAX_MODEL = True
PLOT_RES = True
MONITORING = False
UPDATE_INTERVAL = 5  # every x Epochs
CONF_MATRIX = False

# Training Loop
BATCH_SIZE = 16  # 32, 64
EPOCHS = 120
LR_WARUMUP = True
COSINE = True
SAMPLE_RATE = 22050

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_JSON_PATH = "speechData.json"
MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_JSON_PATH = "musicData.json"

# Spectrogram
NFFT = 1024
HOPLENGTH = 512
NMELS = 128
FMIN = 20
FMAX = 8300
TOPDB = 80
