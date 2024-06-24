# Just run SETS.fit and pickle dump
import argparse
import os

from tqdm import trange

from AE_train import generate_AEs_given_data
from utils.data_util import get_UCR_UEA_sets, get_result_JSON, save_result_JSON

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tslearn.datasets import UCR_UEA_datasets
import pickle
import numpy as np
import torch

import pickle
import CFs


def SETS_mine(dataset, max_len=None, contract_time_per_dim=None, UCR_UEA_dataloader=None):
    shapelets_path = f'../shapelets/{dataset}'
    if UCR_UEA_dataloader is None:
        UCR_UEA_dataloader = UCR_UEA_datasets()
    if not os.path.exists(shapelets_path):
        os.makedirs(shapelets_path)
    X_train, train_y, X_test, test_y = UCR_UEA_dataloader.load_dataset(dataset)
    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])

    model = None  # this is merely for the TSInterpert structure
    TS_feature_num = X_train.shape[-1]
    TS_length = X_train.shape[-2]
    print(TS_feature_num, TS_length)
    if max_len is None:
        max_len = int(TS_length / 2)
    if contract_time_per_dim is None:
        contract_time_per_dim = int(240 / TS_feature_num)
    SETS = CFs.SETSCF(model,
                      (train_x, train_y),
                      backend='PYT',
                      mode='feat',
                      min_shapelet_len=3,
                      max_shapelet_len=max_len,
                      time_contract_in_mins_per_dim=contract_time_per_dim,
                      fit_shapelets=False)
    SETS.fit(occlusion_threshhold=1e-1, remove_multiclass_shapelets=True)
    with open(os.path.join(shapelets_path, 'SETS.pkl'), 'wb') as file:
        pickle.dump(SETS, file)
    return "Fitted"


def mining_shapelets_for_SETS(dataset_choice, start_per: float = 0.0, end_per: float = 1.0):
    datasets = get_UCR_UEA_sets(dataset_choice)
    model_name = 'Shapelets'
    UCR_UEA_dataloader = UCR_UEA_datasets()

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
            best_train_loss = SETS_mine(dataset, max_len=None, contract_time_per_dim=None, UCR_UEA_dataloader=UCR_UEA_dataloader)

            method_record = get_result_JSON(model_name)
            method_record[dataset] = best_train_loss
            save_result_JSON(method_name=model_name, record_dict=method_record)

        else:
            acc = method_record.get(dataset)
            print(f'{model_name} on {dataset} was already fitted')

        pbar.update(1)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Specify models and datasets')
    # parser.add_argument('--dataset_choice', type=str, default='all', help='dataset name')
    # parser.add_argument('--start_per', type=float, default=0.0, help='starting percentage of whole datasets')
    # parser.add_argument('--end_per', type=float, default=1.0, help='ending percentage of whole datasets')
    # args = parser.parse_args()
    # mining_shapelets_for_SETS(args.dataset_choice, args.start_per, args.end_per)
    SETS_mine('FaceDetection', max_len=None, contract_time_per_dim=None, UCR_UEA_dataloader=UCR_UEA_datasets())