HYPERPARAMS_SPEECH = {
    'learning_rate': [1e-4],
    'decay_epoch': [40],  # every x epochs LR will be updated
    'decay_rate': [2],  # LR will be updated to LR / x
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [3],
    'init_stride': [2]
}

HYPERPARAMS_ESC50 = {
    'learning_rate': [1e-4, 2e-4],
    'decay_epoch': [20, 40],  # every x epochs LR will be updated
    'decay_rate': [2, 4, 6],  # LR will be updated to LR / x
    'rrc': [True],  # RandomResizeCrop
    'rlf': [False],  # RandomLinearFade
    'dual_patchnorm': [True],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [(3, 3)],
    'init_stride': [(2, 2)]
}

HYPERPARAMS_MUSIC = {
    'learning_rate': [1e-4],
    'decay_epoch': [20],  # every x epochs LR will be updated
    'decay_rate': [1.2],  # LR will be updated to LR / x
    'rrc': [True],  # RandomResizeCrop
    'rlf': [True],  # RandomLinearFade
    'dual_patchnorm': [False],
    'mixup': [True],  # only 1 value possible atm
    'init_kernel_size': [8, 4],
    'init_stride': [4, 2]
}

# General
DATASET = "SPEECH"  # "SPEECH", "ESC50", "MUSIC"
PLOT_RES = True
MONITORING = False
UPDATE_INTERVAL = 1  # every x Epochs
ONLY_TABULATE = False
CONF_MATRIX = False

# Training Loop
BATCH_SIZE = 16  # 32
EPOCHS = 20
LR_SCHEDULING = True
SAMPLE_RATE = 22050  # 16k

# Dataset paths
SPEECH_DATASET_PATH = "data/SpeechCommands"
SPEECH_JSON_PATH = "speechData.json"
MUSIC_DATASET_PATH = "data/GTZAN/genres_original"
MUSIC_JSON_PATH = "musicData.json"

# Spectrogram
NFFT = 1024
HOPLENGTH = 512
NMELS = 64  # 128
FMIN = 20
FMAX = 8300
TOPDB = 80

# Augmentations
# TIME_MASKING = False
# FREQ_MASKING = False
# TIME_SHIFT_ON = False
# NOISE_INJECT = False  # zu stark, eh useless
