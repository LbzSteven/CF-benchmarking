import os
import time
from datetime import date
import pandas as pd

from utils.data_util import read_UCR_UEA
from utils.model_util import model_init
from utils.visual_util import visualize_TSinterpret

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tslearn.datasets import UCR_UEA_datasets
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
import pickle

from models.ResNet import ResNetBaseline
from models.FCN import FCN
from utils.train_util import get_all_preds, UCRDataset, generate_loader
from utils.model_util import get_AE_dict
from quantative.metric import proximity, sparsity, mean_absolute_error, plausibility, generate_metric_stat

warnings.filterwarnings("ignore")
# from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
import CFs


def CF_generate(dataset, model_name, CF_method='NG', AE_name='FCN_AE', vis_flag=False):
    model_dataset_path = f'../models/{model_name}/{dataset}'
    CF_path = f'../CF_result/{CF_method}/{model_name}/{dataset}'
    print(f'Generating {CF_path} counterfactual on {model_name} in {dataset} dataset')

    if os.path.exists(CF_path):
        print(f'{CF_path} exist')
    else:
        os.makedirs(CF_path)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset)

    train_loader, test_loader = generate_loader(train_x, test_x, train_y, test_y)

    in_channels = train_x.shape[-2]
    input_size = train_x.shape[-1]
    model = model_init(model_name, in_channels=in_channels, n_pred_classes=train_y.shape[1])
    state_dict = torch.load(f'{model_dataset_path}/weight.pt')
    model.load_state_dict(state_dict)
    model.eval()
    y_pred = np.load(f'{model_dataset_path}/test_preds.npy')

    if CF_method == 'NG':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_y), backend='PYT', mode='feat', method='NUN_CF', max_iter=10000)
        # Now TSInterpret is swapped
    elif CF_method == 'NUN_CF':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_y), backend='PYT', mode='feat', method='NG', max_iter=10000)
    elif CF_method == 'NG_DTW':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_y), backend='PYT', mode='feat', method='dtw_bary_center',
                                       max_iter=10000)
    elif CF_method == 'TSEvo':
        exp_model: TSEvo = TSEvo(model=model.cpu(), data=(test_x, np.array(y_pred)), mode='feat', backend='PYT', epochs=500)
        # TODO this is so weird, check if it is correct
    elif CF_method == 'COMTE':
        exp_model: COMTECF = COMTECF(model, (train_x, train_y), backend='PYT', mode='feat', method='opt')
    elif CF_method == 'SETS':
        # exp_model = CFs.SETSCF(model, (train_x, train_y), backend='PYT', mode='feat', method='opt')
        shapelets_path = f'../shapelets/{dataset}/SETS.pkl'
        if os.path.exists(shapelets_path):
            with open(shapelets_path, 'rb') as file:
                exp_model: CFs.SETSCF = pickle.load(file)
            exp_model.set_models(model)
        else:
            raise 'mine shapelet for SETS first'
    elif CF_method == 'wCF':
        exp_model: CFs.wCF = CFs.wCF(model, train_x, backend='PYT', mode='feat', max_iter=500, lambda_init=10, pred_threshold=0.5)
    else:
        raise 'Undefined CF method'

    # initialize metrics:
    AE_path = f'../models/AE/{dataset}'
    AE_dict = get_AE_dict(AE_name, AE_path, in_channels, input_size)

    exp_results = []
    L0 = []
    L1 = []
    L2 = []
    Linf = []
    maes = []
    IM1 = []
    AE_loss = []
    criterion = nn.MSELoss()
    num_valid = 0

    list_valid = []
    generation_times = []

    num_instance = test_x.shape[0]

    # Fro TSEvo
    np.random.seed(42)
    random_selection = np.random.choice(num_instance, size=20, replace=False)

    for i in range(num_instance):
        orig = test_x[i].reshape(1, in_channels, -1)
        pred_label = y_pred[i]  # The pred_label is the True label for TSEvo
        if isinstance(exp_model, TSEvo):
            random_i = random_selection[i]
            orig = test_x[random_i].reshape(1, in_channels, -1)
            pred_label = y_pred[random_i]  # The pred_label is the True label for TSEvo
        start = time.time()

        if isinstance(exp_model, COMTECF) or isinstance(exp_model, CFs.SETSCF) or isinstance(exp_model, CFs.wCF):  # we keep the target as the second largest prediction
            CF, pred_CF = exp_model.explain(orig)
        else:
            CF, pred_CF = exp_model.explain(orig, pred_label)  # Check again for NUNCF

        durations = time.time() - start
        generation_times.append(durations)
        if isinstance(exp_model, TSEvo):
            pred_CF = np.argmax(pred_CF)
        if pred_CF is not None:
            orig = orig.reshape(in_channels, -1)
            CF = CF.reshape(in_channels, -1)
            exp_results.append(CF)
            L0.append(sparsity(orig, CF))
            L1_i, L2_i, Linf_i = proximity(orig, CF)

            IM1_i, _, _, _, AE_loss_i = plausibility(pred_CF, pred_label, AE_dict, CF, criterion)
            L1.append(L1_i)
            L2.append(L2_i)
            Linf.append(Linf_i)
            maes.append(mean_absolute_error(orig, CF))
            list_valid.append(i)
            IM1.append(IM1_i)
            AE_loss.append(AE_loss_i)
            if i <= 49 and vis_flag:
                marker = i if not isinstance(exp_model, TSEvo) else random_selection[i]
                visualize_TSinterpret(exp_model, orig, pred_label, CF, pred_CF, CF_path, marker)
            num_valid = num_valid + 1

        if isinstance(exp_model, TSEvo) and i == 19:  # TSEvo only execute 20 random times
            break
    np.save(f'{CF_path}/CF.npy', np.array(exp_results))
    np.save(f'{CF_path}/valid.npy', np.array(list_valid))
    return generate_metric_stat(L0, L1, L2, Linf, maes, IM1, AE_loss, generation_times, num_instance, num_valid)


def get_all_CF():
    uni = UCR_UEA_datasets().list_univariate_datasets()
    mul = UCR_UEA_datasets().list_multivariate_datasets()
    all = UCR_UEA_datasets().list_datasets()  # todo check if all the dataset are without missing data and are same length

    datasets = ['GunPoint']
    # datasets = ['CBF', 'Coffee', 'ElectricDevices', 'ECG5000', 'GunPoint', 'FordA', 'Heartbeat', 'PenDigits',
    #             'UWaveGestureLibrary', 'NATOPS']
    # model_names = ['ResNet', 'FCN']
    model_names = ['FCN']
    # CF_methods = ['NG', 'NUN_CF', 'NG_DTW', 'TSEvo']
    # CF_methods = ['SETS']
    CF_methods = ['wCF']
    metric_list = ['L0', 'L0_std', 'L1', 'L1_std', 'L2', 'L2_std', 'Linf', 'Linf_std', 'maes', 'maes_std', 'IM1', 'IM1_std', 'AEloss', 'AEloss_std', 'gtime', 'gtime_std',
                   'valid']  # TODO put this thing in JSON
    df = pd.DataFrame(columns=['CF', 'model', 'dataset'] + metric_list)

    for CF_method in CF_methods:
        for model_name in model_names:
            for dataset in datasets:
                row = [CF_method, model_name, dataset] + CF_generate(dataset, model_name, CF_method, vis_flag=True)
                df.loc[len(df.index)] = row
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].round(3)
    df.to_csv(f'../Summary/CF/all_CF_{date.today()}.csv')


if __name__ == '__main__':
    get_all_CF()
