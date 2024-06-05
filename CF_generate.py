import argparse
import os
import time
from datetime import date
import pandas as pd
from tqdm import trange

from utils.data_util import read_UCR_UEA, get_UCR_UEA_sets, save_result_JSON, get_result_JSON
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
# from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
# from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
import CFs


def CF_generate(dataset, model_name, CF_method='NG', AE_name='FCN_AE', vis_flag=False, device='cuda:0'):
    model_dataset_path = f'../models/{model_name}/{dataset}'
    CF_path = f'../CF_result/{CF_method}/{model_name}/{dataset}'
    print(f'Generating {CF_path} counterfactual on {model_name} in {dataset} dataset')

    if os.path.exists(CF_path):
        print(f'{CF_path} exist')
    else:
        os.makedirs(CF_path)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())

    _, train_loader_no_shuffle = generate_loader(train_x, train_x, train_y, train_y)

    in_channels = train_x.shape[-2]
    input_size = train_x.shape[-1]
    n_pred_classes = train_y.shape[1]
    model = model_init(model_name, in_channels=in_channels, n_pred_classes=n_pred_classes, seq_len=input_size)
    state_dict = torch.load(f'{model_dataset_path}/weight.pt')
    model.load_state_dict(state_dict)
    model.eval()
    test_pred = np.load(f'{model_dataset_path}/test_preds.npy')
    train_pred, _ = get_all_preds(model, train_loader_no_shuffle, device=device)
    train_pred = np.array(train_pred)

    if CF_method == 'NG':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_pred), backend='PYT', mode='feat', method='NG', max_iter=input_size, device=device) # Here we use train_pred otherwise may generate invalid prediction
    elif CF_method == 'NUN_CF':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_pred), backend='PYT', mode='feat', method='NUN_CF', max_iter=input_size, device=device)
    elif CF_method == 'NG_DTW':
        exp_model: CFs.NGCF = CFs.NGCF(model, (train_x, train_pred), backend='PYT', mode='feat', method='dtw_bary_center',
                                       max_iter=10000, device=device)
    elif CF_method == 'TSEvo':
        exp_model: CFs.TSEvo = CFs.TSEvo(model=model, data=(test_x, np.array(test_pred)), mode='feat', backend='PYT', epochs=500, device=device)
    elif CF_method == 'COMTE':
        exp_model: CFs.COMTECF = CFs.COMTECF(model, (train_x, train_pred), backend='PYT', mode='feat', method='opt', device=device)
    elif CF_method == 'SETS':
        # exp_model = CFs.SETSCF(model, (train_x, train_y), backend='PYT', mode='feat', method='opt')
        shapelets_path = f'../shapelets/{dataset}/SETS.pkl'
        if os.path.exists(shapelets_path):
            with open(shapelets_path, 'rb') as file:
                exp_model: CFs.SETSCF = pickle.load(file)
            exp_model.set_models(model, device=device)
        else:
            raise 'mine shapelets for SETS first'
    elif CF_method == 'wCF':
        exp_model: CFs.wCF = CFs.wCF(model, train_x, backend='PYT', mode='feat', max_iter=500, lambda_init=10, pred_threshold=0.5, device=device)
    else:
        raise 'Undefined CF method'

    # initialize metrics:
    # AE_path = f'../models/AE/{AE_name}/{dataset}'
    # AE_dict = get_AE_dict(AE_name, AE_path, in_channels, input_size)

    exp_results = []
    pred_CFs = []
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
    pbar = trange(num_instance, desc='Dataset', unit='epoch', initial=0, disable=False)
    for i in range(num_instance):
        orig = test_x[i].reshape(1, in_channels, -1)
        pred_label = test_pred[i]  # The pred_label is the True label for TSEvo
        if isinstance(exp_model, CFs.TSEvo):
            random_i = random_selection[i]
            orig = test_x[random_i].reshape(1, in_channels, -1)
            pred_label = int(test_pred[random_i])  # The pred_label is the True label for TSEvo

        start = time.time()

        if isinstance(exp_model, CFs.COMTECF) or isinstance(exp_model, CFs.SETSCF) or isinstance(exp_model, CFs.wCF):  # we keep the target as the second largest prediction
            CF, pred_CF = exp_model.explain(orig)
        else:
            CF, pred_CF = exp_model.explain(orig, pred_label)

        durations = time.time() - start
        generation_times.append(durations)
        if isinstance(exp_model, CFs.TSEvo):
            pred_CF = np.argmax(pred_CF)
        if pred_CF is not None:
            orig = orig.reshape(in_channels, -1)
            CF = CF.reshape(in_channels, -1)

            L1_i, L2_i, Linf_i = proximity(orig, CF)

            # IM1_i, _, _, _, AE_loss_i = plausibility(pred_CF, pred_label, AE_dict, CF, criterion)

            exp_results.append(CF)
            pred_CFs.append(pred_CF)
            L0.append(sparsity(orig, CF))
            L1.append(L1_i)
            L2.append(L2_i)
            Linf.append(Linf_i)
            maes.append(mean_absolute_error(orig, CF))
            list_valid.append(i)
            # IM1.append(IM1_i)
            # AE_loss.append(AE_loss_i)
            if i <= 49 and vis_flag:
                marker = i if not isinstance(exp_model, CFs.TSEvo) else random_selection[i]
                visualize_TSinterpret(exp_model, orig, pred_label, CF, pred_CF, CF_path, marker)
            num_valid = num_valid + 1
        pbar.update(1)
        if isinstance(exp_model, CFs.TSEvo) and i == 19:  # TSEvo only execute for 20 random instances
            break
    np.save(f'{CF_path}/CF.npy', np.array(exp_results))
    np.save(f'{CF_path}/valid.npy', np.array(list_valid))
    np.save(f'{CF_path}/pred_CFs.npy', np.array(pred_CFs))
    np.save(f'{CF_path}/test_x.npy', test_x)
    np.save(f'{CF_path}/test_pred.npy', test_pred)
    np.save(f'{CF_path}/test_y.npy', np.argmax(test_y, axis=1))  # Change from onehot to a class number notice this is different from the original class name
    np.save(f'{CF_path}/generation_times.npy', np.array(generation_times))
    return generate_metric_stat(L0, L1, L2, Linf, maes, generation_times, num_instance, num_valid)


# def get_all_CF():
#     uni = UCR_UEA_datasets().list_univariate_datasets()
#     mul = UCR_UEA_datasets().list_multivariate_datasets()
#     all = UCR_UEA_datasets().list_datasets()
#
#     datasets = ['BasicMotions']
#     # datasets = ['CBF', 'Coffee', 'ElectricDevices', 'ECG5000', 'GunPoint', 'FordA', 'Heartbeat', 'PenDigits',
#     #             'UWaveGestureLibrary', 'NATOPS']
#     # model_names = ['ResNet', 'FCN']
#     model_names = ['FCN']
#     # CF_methods = ['NG', 'NUN_CF', 'NG_DTW', 'TSEvo']
#     # CF_methods = ['SETS']
#     CF_methods = ['SETS', 'wCF', 'NUN_CF'] # TODO test them on basic motions CF_methods = ['SETS', 'wCF', 'NUN_CF']
#     metric_list = ['L0', 'L0_std', 'L1', 'L1_std', 'L2', 'L2_std', 'Linf', 'Linf_std', 'maes', 'maes_std', 'IM1', 'IM1_std', 'AEloss', 'AEloss_std', 'gtime', 'gtime_std',
#                    'valid']
#     df = pd.DataFrame(columns=['CF', 'model', 'dataset'] + metric_list)
#
#     for CF_method in CF_methods:
#         for model_name in model_names:
#             for dataset in datasets:
#                 row = [CF_method, model_name, dataset] + CF_generate(dataset, model_name, CF_method, vis_flag=True)
#                 df.loc[len(df.index)] = row
#     numeric_columns = df.select_dtypes(include=['number']).columns
#     df[numeric_columns] = df[numeric_columns].round(3)
#     df.to_csv(f'../Summary/CF/all_CF_{date.today()}.csv')

def generate_CF(CF_name, model_name, dataset_choice, device: str = 'cuda:0', start_per: float = 0.0, end_per: float = 1.0):
    datasets = get_UCR_UEA_sets(dataset_choice)
    UCR_UEA_dataloader = UCR_UEA_datasets()
    if model_name == 'all':
        models = ['ResNet', 'FCN', 'InceptionTime', 'MLP']
    else:
        models = [model_name]

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
            CF_classification = 'CF/'+model_name+'/'+CF_name
            dataset = datasets[i]
            pbar.set_postfix(loss=f'{dataset}')
            method_record = get_result_JSON(CF_classification)
            # get reference acc of this method on this dataset
            print(f'Generating CF with {CF_name} on {dataset} with model {model_name}')
            results = method_record[dataset]
            if 'NotEvaluate' in results:
                # acc = train_model_datasets(dataset, model_name, device, UCR_UEA_dataloader)
                # acc = np.random.randint(1)
                results = CF_generate(dataset, model_name, CF_name, vis_flag=False, device=device)
                method_record = get_result_JSON(CF_classification)
                method_record[dataset] = results
                save_result_JSON(method_name=CF_classification, record_dict=method_record)
            else:

                print(f'CFs of {model_name} on {dataset} was already generated by {CF_name} with {results}')

            pbar.update(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify classification and counterfactual models and datasets')
    parser.add_argument('--CF_name', type=str, default='NG', help='classification model name')
    parser.add_argument('--model_name', type=str, default='FCN', help='classification model name')
    parser.add_argument('--dataset_choice', type=str, default='uni', help='dataset name')
    parser.add_argument('--CUDA', type=str, default='cuda:0', help='CUDA')
    parser.add_argument('--start_per', type=float, default=0.0, help='starting percentage of whole datasets')
    parser.add_argument('--end_per', type=float, default=1.0, help='ending percentage of whole datasets')
    args = parser.parse_args()
    generate_CF(args.CF_name, args.model_name, args.dataset_choice, args.CUDA, args.start_per, args.end_per)
    # for CF_method in ['NUN_CF']:
    #     # for CF_method in ['NG']:
    #     model_name = 'MLP'
    #     dataset = 'BasicMotions'
    #     CF_generate(dataset, model_name, CF_method=CF_method, AE_name='FCN_AE', vis_flag=True)
    # for CF_method in ['NUN_CF']:
    #     # for CF_method in ['NG']:
    #     model_name = 'MLP'
    #     dataset = 'Computers'
    #     CF_generate(dataset, model_name, CF_method=CF_method, AE_name='FCN_AE', vis_flag=True)
