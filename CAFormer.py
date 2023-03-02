import logging
import time
from itertools import cycle, product
import tqdm
from colorlog import ColoredFormatter

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from x_metaformer import CAFormer, CFFormer

import utilsPlotting
from ESC50Dataset import ESC50Data
from MusicDataset import split_music_dataset, MusicDataset
from SpeechDataset import *
from AudioUtil import *
from metrics import *


# SETTINGS
def settings():
    if DATASET == "SPEECH":
        current_hypers = HYPERPARAMS_SPEECH
    elif DATASET == "MUSIC":
        current_hypers = HYPERPARAMS_MUSIC
    elif DATASET == "ESC50":
        current_hypers = HYPERPARAMS_ESC50
    return current_hypers


def initiate_logging():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    log_level = logging.DEBUG
    log_format = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(log_format)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger('pythonConfig')
    log.setLevel(log_level)
    log.addHandler(stream)
    return log


def get_data(mixup):
    global num_classes, categories, train_data, valid_data
    if DATASET == "SPEECH":
        num_classes = 30
        # preprocess_speech_dataset(SPEECH_DATASET_PATH, SPEECH_JSON_PATH)
        x_train, y_train, x_validation, y_validation, x_test, y_test, categories = split_speech_dataset(
            SPEECH_JSON_PATH)
        # unique, counts = np.unique(y_train, return_counts=True)
        # print(dict(zip(unique, counts)))
        # print(f"categories: {categories}")
        train_data = SpeechDataset("training", x_train, y_train, categories, mixup=mixup)
        valid_data = SpeechDataset("validation", x_validation, y_validation, categories, mixup=False)

    elif DATASET == "ESC50":
        num_classes = 50
        audio_path = "data/ESC-50-master/audio/"
        metadata_file = "data/ESC-50-master/meta/esc50.csv"
        df = pd.read_csv(metadata_file)
        # print(df.head())

        train_df = df[df['fold'] != 5]
        valid_df = df[df['fold'] == 5]

        train_data = ESC50Data("training", audio_path, train_df, 'filename', 'category', mixup=mixup)
        valid_data = ESC50Data("validation", audio_path, valid_df, 'filename', 'category', mixup=False)

    elif DATASET == "MUSIC":
        num_classes = 10
        # preprocess_music_dataset(MUSIC_DATASET_PATH, MUSIC_JSON_PATH)
        x_train, y_train, x_validation, y_validation, x_test, y_test, categories = split_music_dataset(MUSIC_JSON_PATH)
        # print(categories)
        # unique, counts = np.unique(y_train, return_counts=True)
        # print(dict(zip(unique, counts)))
        train_data = MusicDataset("training", x_train, y_train, categories, mixup=mixup)
        valid_data = MusicDataset("validation", x_validation, y_validation, categories, mixup=False)
    return train_data, valid_data, num_classes


def setlr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return opt


def lr_decay(opt, epoch, hyperparameters):
    # new_lr = LEARNING_RATE / (10 ** (epoch // LRDECAY_EPOCH))
    current_learning_rate = opt.param_groups[0]["lr"]
    new_lr = current_learning_rate / hyperparameters[2]
    opt = setlr(opt, new_lr)
    log.info(f'Changed learning rate to {new_lr}')
    return opt


def lr_lambda(epoch):
    if epoch < EPOCHS * 0.1:
        return epoch / (EPOCHS * 0.1)
    else:
        return 1.0


def train_with_hyperparams(hyperparams, filepath):
    # Get all possible combinations of hyperparameters
    hyperparam_values = list(hyperparams.values())
    hyperparam_combinations = list(product(*hyperparam_values))
    log.info(f"Generated {len(hyperparam_combinations)} possible hyperparameter combinations.")
    # Train model with each combination of hyperparameters and save results
    averages = {}
    for i, combo in enumerate(hyperparam_combinations):
        my_metaformer = CAFormer(
            in_channels=1,
            depths=(3, 3, 9, 3),
            dims=(64, 128, 320, 512),
            init_kernel_size=combo[7],
            # init_kernel_size=3,
            init_stride=combo[8],
            # init_stride=2,
            drop_path_rate=0.5,
            norm='ln',  # ln, bn or rms (layernorm, batchnorm or rmsnorm)
            use_dual_patchnorm=combo[5],  # norm on both sides for the patch embedding
            use_pos_emb=True,  # use 2d sinusodial positional embeddings
            head_dim=32,
            num_heads=4,
            attn_dropout=0.1,
            proj_dropout=0.1,
            patchmasking_prob=0,  # replace 5% of the initial tokens with a </mask> token
            scale_value=1.0,  # scale attention logits by this value
            trainable_scale=False,  # if scale can be trained
            num_mem_vecs=0,  # additional memory vectors (in the attention layers)
            sparse_topk=0,  # sparsify - keep only top k values (in the attention layers)
            l2=False,  # l2 norm on tokens (in the attention layers)
            improve_locality=False,  # remove attention on own token
            use_starreglu=False  # use gated StarReLU
        )
        my_metaformer = my_metaformer.to(device)
        log.info(f"Training Cycle {i + 1} of {len(hyperparam_combinations)}")

        training_results = train(my_metaformer, criterion, train_loader, valid_loader, combo, num_classes,
                                 iteration=[i + 1, len(hyperparam_combinations)])

        # calculate the average of the last 5 values for each key
        last_five = training_results[-5:]
        for key in last_five[0].keys():
            averages[key] = np.average([elem[key] for elem in last_five])

        max_acc = np.max([elem['avg_valid_acc'] for elem in training_results])
        log.info(f'Maximum Accuracy: {max_acc}')

        if SAVE_DATA:
            # Append hyperparams and results to json file
            if not os.path.exists(filepath):
                with open(filepath, 'w+') as f:
                    json.dump([], f)

            with open(filepath, 'r') as file:
                data = json.load(file)

            data.append(combo)
            data.append(averages)

            with open(filepath, 'w') as f:
                json.dump(data, f)

        if CONF_MATRIX:
            log.info("Calculating Confusion Matrix")
            conf_matrix(my_metaformer, valid_loader, valid_data)

        if PLOT_RES:
            utilsPlotting.plot_results(training_results)


def train(model, loss_fn, train_loader, val_loader, hyperparameters, classes, iteration):
    start_time = time.time()

    epoch_id = 0
    res_array = []
    train_losses = []
    valid_losses = []
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters[0])
    scheduler_warumup = LambdaLR(optimizer, lr_lambda)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=0.9*EPOCHS)

    for epoch in (range(1, EPOCHS + 1)):
        epoch_id += 1
        log.info(f"Iteration {iteration[0]}/ {iteration[1]} - Epoch {epoch_id} / {EPOCHS}")
        model.train()
        batch_losses = []
        if LR_SCHEDULING and epoch % hyperparameters[1] == 0:  # hyperparameters[1] = decay_epoch
            optimizer = lr_decay(optimizer, epoch, hyperparameters)

        for i, data in tqdm(enumerate(train_loader), desc='Epoch progress', ncols=33):
            x, y = data
            # if RANDOM_RESIZE_CROP:
            if hyperparameters[3]:
                resize_cropper = RandomResizeCrop()
                x = resize_cropper(x)
            # if RANDOM_LINEAR_FADE:
            if hyperparameters[4]:
                linear_fader = RandomLinearFader()
                x = linear_fader(x)
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)  # y.float() for mixup?
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        if LR_WARUMUP and epoch_id < (EPOCHS / 10):
            scheduler_warumup.step()
        if COSINE and epoch_id > (EPOCHS / 10):
            scheduler_cos.step()
        train_losses.append(batch_losses)
        train_loss = np.mean(train_losses[-1])
        # print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        # y_pred = []
        # y_true = []
        batch_losses, trace_y, trace_yhat, lrs = [], [], [], []
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])
            # output = (torch.max(torch.exp(y_hat), 1)[1]).data.cpu().numpy()
            # y_pred.extend(output)  # Save Prediction
            # labels = y.data.cpu().numpy()
            # y_true.extend(labels)  # Save Truth
            # classes = data.categories
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        valid_loss = np.mean(valid_losses[-1])
        f1 = f1score(model, val_loader)
        res_array.append(
            {'avg_valid_loss': valid_loss, 'avg_valid_acc': accuracy, 'avg_train_loss': train_loss,
             'lr': np.average(lrs), 'f1': f1})
        if MONITORING and epoch_id % UPDATE_INTERVAL == 0 and epoch_id != EPOCHS:
            # print(result_array[epoch_id-1])
            utilsPlotting.liveplot(res_array,
                                   [{"Accuracies vs epochs": ['avg_valid_acc']}, {"f1_score vs epochs": ['f1']},
                                    {"Train Losses vs epochs": ['avg_train_loss']},
                                    {"Valid Losses vs epochs": ['avg_valid_loss']},
                                    {"Learning rates per batch vs epochs": ['lr']}], epoch=EPOCHS)
        # print(f'Epoch - {epoch} Valid-Loss : {valid_loss} Valid-Accuracy : {accuracy}')
    log.info(f"Training duration: {(time.time() - start_time)} seconds")
    return res_array


if __name__ == "__main__":
    if ONLY_TABULATE:
        filename = f"results//results_{DATASET}.json"
        utilsPlotting.tabulate_data(filename)
        quit()

    current_hyperparams = settings()
    log = initiate_logging()
    log.info(f'Loading and preprocessing {DATASET} datasets...')

    # plot_example()
    # plt.show()
    # quit()
    train_data, valid_data, num_classes = get_data(current_hyperparams['mixup'][0])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    log.info(f"selected device: {device}")

    # optimizer = optim.Adam(my_metaformer.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, g)
    criterion = nn.CrossEntropyLoss()

    filename = f"results//results_{DATASET}.json"
    train_with_hyperparams(current_hyperparams, filename)

    # CAmodel = CAFormer(
    #     in_channels=1,
    #     depths=(3, 3, 9, 3),
    #     dims=(64, 128, 320, 512),
    #     init_kernel_size=3,
    #     init_stride=2,
    #     drop_path_rate=0.5,
    #     norm='ln',  # ln, bn or rms (layernorm, batchnorm or rmsnorm)
    #     use_dual_patchnorm=False,  # norm on both sides for the patch embedding
    #     use_pos_emb=True,  # use 2d sinusodial positional embeddings
    #     head_dim=32,
    #     num_heads=4,
    #     attn_dropout=0.1,
    #     proj_dropout=0.1,
    #     patchmasking_prob=0.05,  # replace 5% of the initial tokens with a </mask> token
    #     scale_value=1.0,  # scale attention logits by this value
    #     trainable_scale=False,  # if scale can be trained
    #     num_mem_vecs=0,  # additional memory vectors (in the attention layers)
    #     sparse_topk=0,  # sparsify - keep only top k values (in the attention layers)
    #     l2=False,  # l2 norm on tokens (in the attention layers)
    #     improve_locality=False,  # remove attention on own token
    #     use_starreglu=False  # use gated StarReLU
    # )
    # CAmodel = CAmodel.to(device)
    # results = train(CAmodel, criterion, train_loader, valid_loader)
    #
    # # conf_matrix(my_metaformer, valid_loader, valid_data)
    #
    # # Plot results and return max acc
    # accuracies = []
    # for result in results:
    #     accuracies.append(result['avg_valid_acc'])
    # log.info(f'Maximum Accuracy: {max(accuracies)}')
    #
    # log.info("Plotting results")
    # utilsPlotting.liveplot(results, [{"Accuracies vs epochs": ['avg_valid_acc']}, {"f1_score vs epochs": ['f1']},
    #                                  {"Train Losses vs epochs": ['avg_train_loss']},
    #                                  {"Valid Losses vs epochs": ['avg_valid_loss']},
    #                                  {"Learning rates per batch vs epochs": ['lr']}], epoch=EPOCHS)
    #
    # # with open('esc50resnet.pth', 'wb') as f:
    # #     torch.save(my_metaformer, f)
