import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
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

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = data.categories
    # print(len(classes))

    # Flatten y_true and y_pred
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(20, 14))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')


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
