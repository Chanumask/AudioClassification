import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def conf_matrix(model, loader, data):
    y_pred = []
    y_true = []
    for inputs, labels in loader:
        output = model(inputs.to(device))  # Feed Network

        for i in range(len(output)):
            pred_label = torch.max(torch.exp(output[i]), 0)[1].item()  # Get predicted label
            y_pred.append(pred_label)  # Save Prediction

            true_label = labels[i].item()  # Get true label
            y_true.append(true_label)  # Save Truth

    classes = data.categories
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix.shape)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    # print(f"sum_of_values = {df_cm.values.sum()}")
    plt.figure(figsize=(20, 14))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('results/confusion_matrix.png')


def f1score(model, loader):
    y_pred = []
    y_true = []
    for inputs, labels in loader:
        output = model(inputs.to(device))  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return f1


def get_max_acc(training_results):
    accs = []
    for r in training_results:
        accs.append(r['avg_valid_acc'])
    return max(accs)


def uar_score(trace_y, trace_yhat):
    y_pred = trace_yhat.argmax(axis=1)
    y_true = trace_y
    uar = balanced_accuracy_score(y_true, y_pred)
    return uar
