import argparse
import os

import numpy as np
import torch
from tqdm import trange
from tslearn.datasets import UCR_UEA_datasets

from utils.data_util import read_UCR_UEA, get_UCR_UEA_sets, get_result_JSON, save_result_JSON
from utils.model_util import AE_init
from utils.train_util import generate_loader, fit_AE


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

    best_train_loss = fit_AE(AE, train_loader, device)
    torch.save(AE.state_dict(), f'{model_dataset_path}/AE')
    _train_y = np.argmax(train_y, axis=1)

    for c in np.unique(_test_y):
        train_x_c = train_x[np.where(_train_y == c)]
        train_y_c = train_y[np.where(_train_y == c)]

        train_loader_c, _ = generate_loader(train_x_c, train_y_c, train_x_c, train_y_c)

        AEc = AE_init(model_name, in_channels=train_x.shape[-2], input_size=train_x.shape[-1])
        _ = fit_AE(AEc, train_loader_c, device)
        torch.save(AEc.state_dict(), f'{model_dataset_path}/AE_{c}')
    return best_train_loss
def train_AEs(model_name, dataset_choice, device: str = 'cuda:0', start_per: float = 0.0, end_per: float = 1.0):
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
    # model_name = 'AE'

    total_length = len(datasets)
    start = int(start_per * total_length)
    end = int(end_per * total_length)
    print(f'total dataset length:{total_length}')
    print(f'starting:{start},ending:{end}')

    pbar = trange(end - start, desc='Dataset', unit='epoch', initial=0, disable=False)
    for i in range(start, end):
        dataset = datasets[i]
        pbar.set_postfix(loss=f'{dataset}')
        method_record = get_result_JSON(model_name)

        if method_record[dataset] == 'NotTrained':
            best_train_loss = generate_AEs_given_data(dataset, model_name, device, UCR_UEA_dataloader)
            # acc = np.random.randint(1)
            method_record = get_result_JSON(model_name)
            method_record[dataset] = best_train_loss
            save_result_JSON(method_name=model_name, record_dict=method_record)

        else:
            acc = method_record.get(dataset)
            print(f'{model_name} on {dataset} was already trained with acc: {acc:.2f}')

        pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sepcify models and datasets')
    parser.add_argument('--model_name', type=str, default='AE', help='model name')
    parser.add_argument('--dataset_choice', type=str, default='all', help='dataset name')
    parser.add_argument('--CUDA', type=str, default='cuda:0', help='CUDA')
    parser.add_argument('--start_per', type=float, default=0.0, help='starting percentage of whole datasets')
    parser.add_argument('--end_per', type=float, default=1.0, help='ending percentage of whole datasets')
    args = parser.parse_args()
    train_AEs(args.model_name, args.dataset_choice, args.CUDA, args.start_per, args.end_per)