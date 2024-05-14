from datetime import date
import pickle
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from tslearn.datasets import UCR_UEA_datasets

from models.ResNet import ResNetBaseline
from models.FCN import FCN
from utils.model_util import model_init, AE_init

from utils.train_util import fit, get_all_preds, generate_loader, fit_AE
from utils.data_util import read_UCR_UEA

import warnings

warnings.filterwarnings("ignore")


def train_model_datasets(dataset: str, model_name: str, device=torch.device("cpu")):
    # data preprocessing
    model_dataset_path = f'../models/{model_name}/{dataset}'
    print(f'training {model_name} on {dataset}')
    if os.path.exists(model_dataset_path):
        print(f'{model_dataset_path} exist')
    else:
        os.makedirs(model_dataset_path)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset)

    pickle.dump(enc1, open(f'{model_dataset_path}/OneHotEncoder.pkl', 'wb'))

    train_loader, test_loader = generate_loader(train_x, test_x, train_y, test_y)

    model = model_init(model_name, in_channels=train_x.shape[-2], n_pred_classes=train_y.shape[1])
    # train model
    fit(model, train_loader, device=device)
    torch.save(model.state_dict(), f'{model_dataset_path}/weight.pt')

    test_preds, ground_truth = get_all_preds(model, test_loader, device=device)
    ground_truth = np.argmax(ground_truth, axis=1)
    # save prediction
    np.save(f'{model_dataset_path}/test_preds.npy', np.array(test_preds))
    acc = accuracy_score(ground_truth, test_preds)
    print(f'model: {model_name} acc:{acc:.3f}')
    # confusion matrix heatmap
    sns.set(rc={'figure.figsize': (5, 4)})
    heatmap = confusion_matrix(ground_truth, test_preds)
    sns.heatmap(heatmap, annot=True)
    plt.savefig(f'{model_dataset_path}/confusion_matrix.png')
    plt.close()
    # classification report
    a = classification_report(ground_truth, test_preds, output_dict=True)
    dataframe = pd.DataFrame.from_dict(a)
    dataframe.to_csv(f'{model_dataset_path}/classification_report.csv', index=False)

    return acc


# TODO split data with multiple classes and train AEs based on those classes.
def generate_AEs_given_data(dataset: str, model_name: str = 'FCN_AE', device=torch.device("cpu")):
    model_dataset_path = f'../models/AE/{dataset}'
    print(f'training AEs on {dataset}')
    if os.path.exists(model_dataset_path):
        print(f'{model_dataset_path} exist')
    else:
        os.makedirs(model_dataset_path)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset)
    _test_y = np.argmax(test_y, axis=1)
    train_loader, test_loader = generate_loader(train_x, test_x, train_y, test_y)
    AE = AE_init(model_name, in_channels=train_x.shape[-2], input_size=train_x.shape[-1])

    fit_AE(AE, train_loader, device)
    torch.save(AE.state_dict(), f'{model_dataset_path}/AE')
    _train_y = np.argmax(train_y, axis=1)

    for c in np.unique(_test_y):
        train_x_c = train_x[np.where(_train_y == c)]
        train_y_c = train_y[np.where(_train_y == c)]

        train_loader_c, _ = generate_loader(train_x_c, train_y_c, train_x_c, train_y_c)

        AEc = AE_init(model_name, in_channels=train_x.shape[-2], input_size=train_x.shape[-1])
        fit_AE(AEc, train_loader_c, device)
        torch.save(AEc.state_dict(), f'{model_dataset_path}/AE_{c}')


def train_UCR_UEC():
    uni = UCR_UEA_datasets().list_univariate_datasets()
    mul = UCR_UEA_datasets().list_multivariate_datasets()
    all = UCR_UEA_datasets().list_datasets()  # todo check if all the dataset are without missing data and are same length
    print(uni,mul,all)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # A CUDA device object

    else:
        device = torch.device("cpu")  # A CPU device object
    # datasets = ['GunPoint']
    datasets = ['CBF', 'Coffee', 'ElectricDevices', 'ECG5000', 'GunPoint', 'FordA', 'Heartbeat', 'PenDigits',
                'UWaveGestureLibrary', 'NATOPS']
    model_names = ['ResNet', 'FCN']

    dataset_names = []
    m_names = []
    accs = []
    for model_name in model_names:
        for dataset in datasets:
            acc = train_model_datasets(dataset, model_name, device)
            accs.append(acc)
            m_names.append(model_name)
            dataset_names.append(dataset)
    df = pd.DataFrame({'dataset': dataset_names, 'model_name': m_names, 'acc': accs})
    df.to_csv(f'../Summary/classification/all_acc_{date.today()}.csv')


if __name__ == '__main__':

    # datasets = ['Coffee']
    # datasets = ['CBF', 'Coffee', 'ElectricDevices', 'ECG5000', 'GunPoint', 'FordA', 'Heartbeat', 'PenDigits',
    #             'UWaveGestureLibrary', 'NATOPS']
    # for dataset in datasets:
    #     generate_AEs_given_data(dataset=dataset, device=torch.device("cuda"))

    train_UCR_UEC()