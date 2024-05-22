# Just run SETS.fit and pickle dump

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tslearn.datasets import UCR_UEA_datasets
import pickle
import numpy as np
import torch

import pickle
import CFs


def SETS_mine(dataset, max_len=None, contract_time_per_dim=None):
    shapelets_path = f'../shapelets/{dataset}'
    if not os.path.exists(shapelets_path):
        os.makedirs(shapelets_path)
    X_train, train_y, X_test, test_y = UCR_UEA_datasets().load_dataset(dataset)
    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])

    model = None  # this is merely for the TSinterpert structure
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
    return SETS


if __name__ == '__main__':
    dataset = 'GunPoint'
    SETS_mine(dataset)
