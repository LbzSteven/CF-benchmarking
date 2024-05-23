import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import trange

from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils.model_util import model_init
from utils.train_util import fit, get_all_preds, generate_loader
from utils.data_util import read_UCR_UEA, get_UCR_UEA_sets, get_result_JSON, save_result_JSON
from datetime import date
import warnings

warnings.filterwarnings("ignore")


def train_model_datasets(dataset: str, model_name: str, device=torch.device("cpu"), UCR_UEA_dataloader=None):
    # data preprocessing
    model_dataset_path = f'../models/{model_name}/{dataset}'
    print(f'training {model_name} on {dataset}')
    if os.path.exists(model_dataset_path):
        print(f'{model_dataset_path} exist')
    else:
        os.makedirs(model_dataset_path)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_dataloader)
    if train_x is None:
        return None
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


def train_UCR_UEA(model_name, dataset_choice, device: str = 'cuda:0', start_per: float = 0.0, end_per: float = 1.0):
    datasets = get_UCR_UEA_sets(dataset_choice)

    UCR_UEA_dataloader = UCR_UEA_datasets()
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")  # A CUDA device object

        else:
            device = torch.device("cpu")  # A CPU device object
    # datasets = ['GunPoint']
    # datasets = ['CBF', 'Coffee', 'ElectricDevices', 'ECG5000', 'GunPoint', 'FordA', 'Heartbeat', 'PenDigits',
    #             'UWaveGestureLibrary', 'NATOPS']
    if model_name == 'all':
        models = ['ResNet', 'FCN', 'InceptionTime']
    else:
        models = [model_name]

    dataset_names = []
    m_names = []
    accs = []
    total_length = len(datasets)
    start = int(start_per * total_length)
    end = int(end_per * total_length)
    print(f'total dataset length:{total_length}')
    print(f'starting:{start},ending:{end}')
    for model_name in models:
        pbar = trange(end - start, desc='Dataset', unit='epoch', initial=0, disable=False)
        # method_record = get_result_JSON(model_name)
        # recorded_dataset = method_record.keys()
        for i in range(start, end):
            dataset = datasets[i]
            pbar.set_postfix(loss=f'{dataset}')
            method_record = get_result_JSON(model_name)

            if method_record[dataset] == 'NotTrained':
                acc = train_model_datasets(dataset, model_name, device, UCR_UEA_dataloader)
                # acc = np.random.randint(1)
                method_record = get_result_JSON(model_name)
                method_record[dataset] = acc
                save_result_JSON(method_name=model_name, record_dict=method_record)
                if acc is not None:
                    accs.append(acc)
                    m_names.append(model_name)
                    dataset_names.append(dataset)
            else:
                acc = method_record.get(dataset)
                print(f'{model_name} on {dataset} was already trained with acc: {acc:.2f}')

            pbar.update(1)
    df = pd.DataFrame({'dataset': dataset_names, 'model_name': m_names, 'acc': accs})
    df.to_csv(f'../Summary/classification/all_acc_{date.today()}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sepcify models and datasets')
    parser.add_argument('--model_name', type=str, default='uni', help='model name')
    parser.add_argument('--dataset_choice', type=str, default='FCN', help='dataset name')
    parser.add_argument('--CUDA', type=str, default='cuda:0', help='CUDA')
    parser.add_argument('--start_per', type=float, default=0.0, help='starting percentage of whole datasets')
    parser.add_argument('--end_per', type=float, default=1.0, help='ending percentage of whole datasets')
    args = parser.parse_args()
    train_UCR_UEA(args.model_name, args.dataset_choice, args.CUDA, args.start_per, args.end_per)
