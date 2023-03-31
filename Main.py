import logging
import statistics
import time
from itertools import product

import pandas as pd
import torch
import tqdm
from IPython.utils import io
from colorlog import ColoredFormatter

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from x_metaformer import CAFormer
import pytorch_lightning as pl
from ema_pytorch import EMA

import plotting
from dataset_classes.ESC50Dataset import *
from dataset_classes.MusicDataset import *
from dataset_classes.SpeechDataset import *
from dataset_classes.PrimatesDataset import *
from AudioUtil import *
from metrics import *


# SETTINGS
def setup_parameters():
    if DATASET == "SPEECH":
        current_hypers = HYPERPARAMS_SPEECH
    elif DATASET == "MUSIC":
        current_hypers = HYPERPARAMS_MUSIC
    elif DATASET == "ESC50":
        current_hypers = HYPERPARAMS_ESC50
    else:
        current_hypers = HYPERPARAMS_PRIMATES
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


def get_data():
    if not MAJORITY_VOTE:
        mixup = dataset_hyperparameters['mixup'][0]
    else:
        mixup = None
    global num_classes, categories, train_data, valid_data
    if DATASET == "SPEECH":
        num_classes = 30
        x_train, y_train, categories = load_speech_dataset(SPEECH_PATH_TRAIN)
        x_valid, y_valid, _ = load_speech_dataset(SPEECH_PATH_VALID)
        train_data = SpeechDataset("training", x_train, y_train, categories, mixup=mixup)
        valid_data = SpeechDataset("validation", x_valid, y_valid, categories, mixup=False)

    elif DATASET == "ESC50":
        num_classes = 50
        audio_path = "data/ESC-50-master/audio/"
        metadata_file = "data/ESC-50-master/meta/esc50.csv"
        df = pd.read_csv(metadata_file)
        # print(df.head())

        train_df = df[df['fold'] != 1]
        valid_df = df[df['fold'] == 1]

        train_data = ESC50Data("training", audio_path, train_df, 'filename', 'category', mixup=mixup)
        valid_data = ESC50Data("validation", audio_path, valid_df, 'filename', 'category', mixup=False)

    elif DATASET == "MUSIC":
        num_classes = 10
        x_train, y_train, categories = load_music_dataset(MUSIC_PATH_TRAIN)
        x_valid, y_valid, _ = load_music_dataset(MUSIC_PATH_VALID)
        train_data = MusicDataset("training", x_train, y_train, categories, mixup=mixup, filtaug=False)
        valid_data = MusicDataset("validation", x_valid, y_valid, categories, mixup=False, filtaug=False)

    elif DATASET == "PRIMATES":
        num_classes = 5

        x_train, y_train, categories = load_primates_dataset(PRIMATES_CSV_TRAIN)
        x_val, y_val, _ = load_primates_dataset(PRIMATES_CSV_VALID)

        train_data = PrimatesDataset("training", x_train, y_train, categories, mixup=mixup)
        valid_data = PrimatesDataset("validation", x_val, y_val, categories, mixup=False)

    return train_data, valid_data, num_classes


def get_data_loaders(data_training, data_validation):
    if DATASET == "PRIMATES":
        y_train = data_training.labels
        y_val = data_validation.labels
        class_count_train = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        class_count_val = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight_train = 1. / class_count_train
        weight_val = 1. / class_count_val
        samples_weight_train = np.array([weight_train[t] for t in y_train])
        samples_weight_val = np.array([weight_val[t] for t in y_val])
        # print(f'sample_weights: {samples_weight}')

        sampler_train = WeightedRandomSampler(weights=samples_weight_train, num_samples=len(samples_weight_train),
                                              replacement=True)
        sampler_val = WeightedRandomSampler(weights=samples_weight_val, num_samples=len(samples_weight_val),
                                            replacement=True)
        training_loader = DataLoader(data_training, batch_size=BATCH_SIZE, sampler=sampler_train)
        validation_loader = DataLoader(data_validation, batch_size=BATCH_SIZE, sampler=sampler_val)

    else:
        training_loader = DataLoader(data_training, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(data_validation, batch_size=BATCH_SIZE, shuffle=True)
    return training_loader, validation_loader


def exclude_from_wt_decay(named_params, weight_decay, skip_list):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
            # print(f"skipped param {name}")
        else:
            params.append(param)
    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    ]


def lr_lambda(epoch):
    if epoch < EPOCHS * 0.1:
        return epoch / (EPOCHS * 0.1)
    else:
        return 1.0


class MyMetaFormer(CAFormer):
    def __init__(self, *args, num_classes, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_head = nn.Linear(self.out_dim, num_classes)

    def forward(self, x, return_embeddings=False):
        out = super().forward(x, return_embeddings)
        out = self.linear_head(out)
        return out


def train_with_hyperparams(hyperparams, classes_count, filepath, current_seed):
    # Get all possible combinations of hyperparameters except seed as it is handled outside this function
    no_seed = list(hyperparams.values())
    del no_seed[8]
    hyperparam_combinations = list(product(*no_seed))
    log.info(f"Generated {len(hyperparam_combinations)} possible hyperparameter combinations for this seed.")
    # Train model with each combination of hyperparameters except the seed and save results
    for i, combo in enumerate(hyperparam_combinations):
        if not USE_MAX_MODEL:
            my_metaformer = MyMetaFormer(
                num_classes=classes_count,
                in_channels=1,
                depths=(3, 3, 9, 3),
                dims=(64, 128, 320, 512),  # 32-256
                init_kernel_size=combo[5],
                # init_kernel_size=4,
                init_stride=combo[6],
                # init_stride=2,
                drop_path_rate=0.5,  # 0, 0.25 worse
                norm='ln',  # ln, bn or rms (layernorm, batchnorm or rmsnorm)
                use_dual_patchnorm=combo[3],  # norm on both sides for the patch embedding
                use_pos_emb=True,  # use 2d sinusodial positional embeddings
                head_dim=32,
                num_heads=8,
                attn_dropout=0.1,  # 0, 0.1, 0.2 no diff
                proj_dropout=0.1,  # 0, 0.1 no diff
                patchmasking_prob=0,  # worse: 0.05 replace 5% of the initial tokens with a </mask> token
                scale_value=1.0,  # scale attention logits by this value
                trainable_scale=False,  # if scale can be trained
                num_mem_vecs=0,  # additional memory vectors (in the attention layers)  # 16,32
                sparse_topk=0,  # sparsify - keep only top k values (in the attention layers)
                l2=False,  # l2 norm on tokens (in the attention layers)
                improve_locality=False,  # remove attention on own token
                use_starreglu=False,  # use gated StarReLU
                use_seqpool=True,
                use_grn_mlp=False  # False
            )
            my_metaformer = my_metaformer.to(device)
        else:
            my_metaformer = torch.load(f'best_performance/best_{DATASET}_model')
            # my_metaformer = torch.load(f'ensemble_models/SPEECH/300/seed6acc9750.pt')
            my_metaformer = my_metaformer.to(device)

        log.info(
            f"Training Cycle {(i) * seed} of {len(hyperparam_combinations) * len(dataset_hyperparameters['seed'])}")

        criterion = nn.CrossEntropyLoss()
        training_results = train(my_metaformer, criterion, train_loader, valid_loader, combo,
                                 iteration=[(i * seed),
                                            len(hyperparam_combinations) * len(dataset_hyperparameters['seed'])])

        max_acc = np.max([elem['avg_valid_acc'] for elem in training_results])
        max_uar = np.max([elem['uar'] for elem in training_results])
        log.info(f'Maximum Accuracy: {max_acc}, Maximum UAR: {max_uar}')

        if SAVE_DATA:
            # Append hyperparams and results to json file
            if not os.path.exists(filepath):
                with open(filepath, 'w+') as f:
                    json.dump([], f)

            with open(filepath, 'r') as file:
                data = json.load(file)

            data.append(combo)
            data.append(max_acc)
            data.append(max_uar)
            data.append(seed)

            with open(filepath, 'w') as f:
                json.dump(data, f)

        if SAVE_MAX_MODEL:
            with open(f'best_performance/highest_acc_{DATASET}', 'r+') as f:
                current_max_acc = float(f.read())
                # print(current_max_acc)
                if max_acc > current_max_acc:
                    f.seek(0)
                    f.write(str(max_acc))
                    f.close()
                    with open(f'best_performance/best_{DATASET}_model', 'wb') as g:
                        torch.save(my_metaformer, g)
                        g.close()

        if SAVE_ENSEMBLE:
            directory = f'ensemble_models/{DATASET}/{ENSEMBLE_NAME}'
            formatted_acc = int(round(max_acc * 100, 2) * 100)  # 4 digit (9583 = 95,83%)
            model_filename = f"seed{current_seed}acc{formatted_acc}.pt"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(f'{directory}/{model_filename}', 'wb') as g:
                torch.save(my_metaformer, g)
                g.close()

        if CONF_MATRIX:
            log.info("Calculating Confusion Matrix")
            conf_matrix(my_metaformer, valid_loader, valid_data)

        if PLOT_RES:
            plotting.plot_results(training_results)


def train(model, loss_fn, train_loader, val_loader, hyperparameters, iteration):
    start_time = time.time()

    res_array = []
    train_losses = []
    valid_losses = []
    params = exclude_from_wt_decay(
        model.named_parameters(),
        hyperparameters[7],
        ["temp", "temperature", "scale", "norm"],
    )
    optimizer = optim.AdamW(params, lr=hyperparameters[0], weight_decay=hyperparameters[7])
    scheduler_warumup = LambdaLR(optimizer, lr_lambda)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=1.2 * EPOCHS)

    # ema = EMA(model, beta=0.99, update_every=10, update_after_step=EPOCHS * 0.7)
    ema = EMA(model, beta=0.99, update_every=10, update_after_step=1)

    for epoch in (range(1, EPOCHS + 1)):
        log.info(f"Iteration {iteration[0]}/ {iteration[1]} - Epoch {epoch} / {EPOCHS}")
        model.train()
        batch_losses = []

        for i, data in tqdm(enumerate(train_loader), desc='Epoch progress', ncols=33):
            x, y = data

            if hyperparameters[1]:  # RANDOM_RESIZE_CROP
                resize_cropper = RandomResizeCrop()
                x = resize_cropper(x)
            if hyperparameters[2]:  # RANDOM_LINEAR_FADE
                linear_fader = RandomLinearFader()
                x = linear_fader(x)
            # if hyperparameters[9]:
            #     x_squeezed = x.squeeze()
            #     x = filt_aug(x_squeezed)
            #     x = x.unsqueeze(1)
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
            if hyperparameters[8]:  # dseed deleted, everything after -1
                ema.update()
        if LR_WARUMUP and epoch < (EPOCHS / 10):  # /20
            scheduler_warumup.step()
        if COSINE and epoch > (EPOCHS / 10):
            scheduler_cos.step()
        train_losses.append(batch_losses)
        train_loss = np.mean(train_losses[-1])
        model.eval()
        batch_losses, trace_y, trace_yhat, lrs = [], [], [], []
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            if hyperparameters[9]:
                y_hat = ema(x)
            else:
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
        f1 = f1score(model, val_loader)
        uar = uar_score(trace_y, trace_yhat)
        res_array.append(
            {'avg_valid_loss': valid_loss, 'avg_valid_acc': accuracy, 'avg_train_loss': train_loss,
             'lr': np.average(lrs), 'f1': f1, 'uar': uar})
        if MONITORING and epoch % UPDATE_INTERVAL == 0:
            plotting.liveplot(res_array,
                              [{"Accuracies": ['avg_valid_acc']}, {"f1_score": ['f1']},
                               {"Train Losses": ['avg_train_loss']},
                               {"Valid Losses": ['avg_valid_loss']},
                               {"Learning rates per batch": ['lr']}, {"Unweighted Average Recall": ['uar']}])
    seconds = (time.time() - start_time)
    minutes = int(seconds // 60)
    remaining_seconds = int(round(seconds % 60, 0))
    log.info(f"Training duration: {minutes} minutes and {remaining_seconds} seconds")
    return res_array


def load_ensemble(models_dir):
    # Load all the trained models from the directory and return them as a list.
    models = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith(".pt"):
            model_path = os.path.join(models_dir, file_name)
            loaded_model = torch.load(model_path)
            models.append(loaded_model)
    return models


# def predict_ensemble(ensemble_models):
#     pl.seed_everything(seed=1)
#     training_data, validation_data, num_categories = get_data()
#     training_loader, validation_loader = get_data_loaders(training_data, validation_data)
#     label_array = None
#     df_dict = {}
#
#     for c, model in enumerate(ensemble_models):
#         # supress the seed log
#         with io.capture_output() as captured:
#             pl.seed_everything(seed=1)
#
#         model.eval()
#         trace_y, trace_yhat = [], []
#         with torch.no_grad():
#             for j, data in enumerate(validation_loader):
#                 x, y = data
#                 x = x.to(device, dtype=torch.float32)
#                 y = y.to(device, dtype=torch.long)
#                 y_hat = model(x)
#                 trace_y.append(y.cpu().detach().numpy())
#                 trace_yhat.append(y_hat.cpu().detach().numpy())
#             trace_y = np.concatenate(trace_y)
#             trace_yhat = np.concatenate(trace_yhat)
#             # accuracy = np.mean(np.argmax(trace_yhat, axis=1) == trace_y)
#             prediction = np.argmax(trace_yhat, axis=1)
#             pred_array = np.array(prediction)
#             if label_array is None:
#                 label_array = np.array(trace_y)
#                 df_dict[f'label_{c}'] = label_array
#             df_dict[f'pred_{c}'] = pred_array
#     ensemble_df = pd.DataFrame(df_dict)
#     return ensemble_df

def predict_ensemble(ensemble_models):
    pl.seed_everything(seed=1)
    training_data, validation_data, num_categories = get_data()
    training_loader, validation_loader = get_data_loaders(training_data, validation_data)
    predictions = []
    labels = []
    df_dict = {}

    trace_y, trace_yhat = [], []
    with torch.no_grad():
        for j, data in enumerate(validation_loader):
            batch_yhat = []

            for c, model in enumerate(ensemble_models):
                model.eval()
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                y_hat = model(x)
                batch_yhat.append(y_hat.cpu().detach().numpy())
            batch_yhat = np.stack(batch_yhat, axis=1)
            soft_voting = batch_yhat.mean(1)
            predictions_batch = soft_voting.argmax(-1)
            predictions.append(predictions_batch)
            labels.append(y.cpu().detach().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    df_dict[f'labels'] = labels
    df_dict[f'predictions'] = predictions
    ensemble_df = pd.DataFrame(df_dict)
    return ensemble_df


# def majority_vote(df):
#     majority_preds = []
#     for i, row in df.iterrows():
#         label = row['label_0']
#         predictions = row.drop('label_0')
#         mode_pred = statistics.mode(predictions)
#         majority_preds.append(mode_pred)
#     df['majority_vote'] = majority_preds
#     acc = (df['label_0'] == df['majority_vote']).sum() / len(df)
#     return acc


if __name__ == "__main__":
    log = initiate_logging()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    log.info(f"selected device: {device}")

    if ONLY_TABULATE:
        filename = f"results//{SAVE_NAME}//{DATASET}.json"
        # filename = f"ensemble_results//results_{DATASET}.json"
        log.info(f"Here are the results ({filename}):")
        plotting.tabulate_data(filename)
        quit()
    if ONLY_PLOT_EXAMPLE:
        plotted_path = plotting.plot_all(DATASET)
        log.info(f"Plotted Data from path: {plotted_path}")
        quit()
    if MAJORITY_VOTE:
        ensemble_path = f"ensemble_models/{DATASET}/{ENSEMBLE_NAME}"
        log.info(f"Majority voting {DATASET} Dataset. Using Ensemble in {ensemble_path}")
        models = load_ensemble(ensemble_path)
        df = predict_ensemble(models)
        print(df.head())
        accuracy = (df['labels'] == df['predictions']).sum() / len(df)
        uar = balanced_accuracy_score(df['labels'], df['predictions'])  # test

        log.info(f"Voting Ensemble's accuracy: {accuracy}")
        quit()

    dataset_hyperparameters = setup_parameters()

    for i, seed in enumerate(dataset_hyperparameters['seed']):
        log.info(
            f"Training {len(dataset_hyperparameters['seed'])} different Seeds. Currently ({i + 1}/{len(dataset_hyperparameters['seed'])})")
        pl.seed_everything(seed=seed)

        log.info(f'Loading and preprocessing {DATASET} datasets...')
        train_data, valid_data, num_classes = get_data()
        train_loader, valid_loader = get_data_loaders(train_data, valid_data)

        filename = f"results//{SAVE_NAME}//{DATASET}.json"
        # filename = f"results//ensemble_results_{DATASET}_{ENSEMBLE_NAME}.json"
        train_with_hyperparams(dataset_hyperparameters, num_classes, filename, current_seed=seed)

    # CAmodel = CAFormer(
    #     in_channels=1,
    #     depths=(2, 2, 6, 2),
    #     dims=(32, 64, 128, 256),
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
