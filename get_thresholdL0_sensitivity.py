import os
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
from tslearn.datasets import UCR_UEA_datasets

from utils.data_util import read_UCR_UEA, get_valid_CF_given_path, get_UCR_UEA_sets, get_result_JSON
from utils.model_util import model_init

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

values_to_exclude = []

def count_total_different_segments(orig:np.array, cf:np.array, min_gap=None,verbose=False):
    if orig.shape != cf.shape:
        raise ValueError("A and B must be of the same shape")

    num_segment = 0
    if min_gap is None:
        min_gap = 0
    else:
        min_gap = max(int(min_gap*orig.shape[1]),1)
    for dim in range(orig.shape[0]):
        differences = orig[dim] != cf[dim]

        gap_count = min_gap + 1

        for i in range(len(differences)):
            is_different = differences[i]
            if is_different:
                condition = gap_count > min_gap if min_gap!=0 else gap_count >= 1
                if condition:

                    num_segment += 1
                    if verbose:
                        print(f'seg {num_segment} start from:{i}')
                gap_count = 0  # Reset gap count when in a different segment
            else:
                if verbose and gap_count ==0:
                    print(f'seg {num_segment} stop at:{i}')
                gap_count += 1


    return num_segment

def threshold_replace(orig, cf, threshold=0.001,verbose=False):

    drange = (np.max(orig) - np.min(orig)) * threshold
    diff = np.abs(orig - cf)
    result = np.where(diff < drange, orig, cf)
    count_tiny_modification = np.sum((diff < drange) & (diff > 0))
    L0_threshold = np.sum(diff >= drange) / (orig.shape[1]*orig.shape[0])
    if verbose:
        print(drange,np.argwhere((diff < drange) & (diff > 0)),np.sum((diff < drange) & (diff > 0)))

    return result, L0_threshold, count_tiny_modification



def threshold_sparsity(orig, cf, cf_pred, model, threshold=0.001,min_gap=0.01):
    cf_modified, L0_threshold, count_tiny_modification = threshold_replace(orig, cf, threshold=threshold)
    sensitivity = 0
    num_segment = count_total_different_segments(orig,cf_modified,None)
    num_segment_modi = count_total_different_segments(orig, cf_modified,min_gap)
    if count_tiny_modification > 0:
        with torch.no_grad():
            pred = model(torch.from_numpy(cf_modified.reshape(1, cf_modified.shape[0], -1)).float().to(device))
            pred = F.softmax(pred)
            pred = pred.cpu().numpy()
            pred_modi = np.argmax(pred, axis=1)[0]
            if pred_modi != cf_pred:
                sensitivity = 1

    return L0_threshold, sensitivity, num_segment, num_segment_modi

def threshold_sparsity_on_dataset(tx_valid, cfs, cf_preds, model, threshold=0.001,min_gap=0.01):
    L0s = []
    sensitivities = []
    num_valid = tx_valid.shape[0]
    in_channels = tx_valid.shape[-2]
    num_segments =[]
    num_segment_modis = []
    for i in range(num_valid):
        orig = tx_valid[i].reshape(in_channels, -1)
        cf = cfs[i].reshape(in_channels, -1)
        cf_pred = cf_preds[i]
        L0_threshold, sens,num_segment,num_segment_modi = threshold_sparsity(orig, cf, cf_pred, model,
                                                                             threshold=threshold,
                                                                             min_gap=min_gap)
        L0s.append(L0_threshold)
        sensitivities.append(sens)
        num_segments.append(num_segment)
        num_segment_modis.append(num_segment_modi)
    return L0s, sensitivities,num_segments,num_segment_modis


def compute_threshold_L0_datasets(model_name,CF_name,datasets_selection,threshold=0.001,min_gap=0.01):
    threL0_results = []
    datasets = get_UCR_UEA_sets(datasets_selection)
    CF_results = 'CF/' + model_name + '/' + CF_name
    record = get_result_JSON(CF_results)
    pbar = trange(len(datasets), desc='Dataset', unit='epoch', initial=0, disable=False)
    for dataset in datasets:
        results = record[dataset]
        if results[-1] == 0.0 or 'Time out 10 consecutive time' in str(results) or 'NotEvaluate' in str(results) or dataset in values_to_exclude:
            pbar.update(1)
            continue

        CF_path = f'../CF_result/{CF_name}/{model_name}/{dataset}/'
        model_dataset_path = f'../models/{model_name}/{dataset}'

        cf_valid, tx_valid, ty_valid, pred_valid, cf, cf_pred, random_selection, num_instance = get_valid_CF_given_path(
            CF_path)

        train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())
        in_channels = train_x.shape[-2]
        input_size = train_x.shape[-1]
        n_pred_classes = train_y.shape[1]
        model = model_init(model_name, in_channels=in_channels, n_pred_classes=n_pred_classes, seq_len=input_size)

        state_dict = torch.load(f'{model_dataset_path}/weight.pt',map_location=torch.device(device))

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        L0s, sensitivities,num_segments,num_segment_modis = threshold_sparsity_on_dataset(tx_valid, cf, cf_pred, model,
                                                                                          threshold=threshold,
                                                                                          min_gap=min_gap)
        thresmarker = str(threshold).split('.')[-1]

        marker= '' if threshold==0.001 else str(threshold)

        # np.save(os.path.join(CF_path, f'L0thre{thresmarker}_{marker}.npy'), np.array(L0s))
        # np.save(os.path.join(CF_path, f'sensitivity{thresmarker}_{marker}.npy'), np.array(sensitivities))
        # marker = 'min_gap_' + str(min_gap) if min_gap is not None else marker
        np.save(os.path.join(CF_path, f'num_segments_{marker}_no_tol.npy'), np.array(num_segments))
        # np.save(os.path.join(CF_path, f'num_segments_modi_{marker}.npy'), np.array(num_segment_modis))
        # if np.sum(sensitivities) !=0:
        #     print(model_name, dataset, np.where(np.array(sensitivities) == 1))
        threL0_results.append([model_name, CF_name, dataset, np.mean(L0s), np.std(L0s), np.mean(sensitivities),np.mean(num_segments),np.std(num_segments),
                               np.mean(num_segment_modis),np.std(num_segment_modis)])
        pbar.update(1)
    return threL0_results

def compute_threshold_L0_models(model_name, CF_names=None, datasets_selection='selected_uni',threshold=0.001,min_gap=0.01):
    if CF_names is None:
        CF_names = ['NUN_CF', 'wCF','NG', 'TSEvo', 'SETS']
    results = []
    for CF_name in CF_names:
        results +=compute_threshold_L0_datasets(model_name, CF_name, datasets_selection,threshold,min_gap)
    df_threL0_sensitivity = pd.DataFrame(results,
                                         columns=['model_name','CF_name', 'dataset_name', 'L0thre', 'L0thre_std',
                                                          'sensitivity','seg','seg_std','seg_modi','seg_modi_std'])

    marker = '' if threshold == 0.001 else str(threshold)
    marker = marker if min_gap is None else marker + '_min_gap' + str(min_gap)
    # old = pd.read_csv(f'../Summary/threL0_{model_name}_selected_mul_{marker}_2024-07-17.csv')
    # df_threL0_sensitivity = pd.concat([old,df_threL0_sensitivity])
    # df_threL0_sensitivity.to_csv(f'../Summary/threL0_{model_name}_{datasets_selection}_{marker}_{date.today()}.csv')

    return df_threL0_sensitivity

if __name__ =='__main__':
    threshold = 0.0025
    min_gap = 0.01
    # datasets_selection = 'selected_uni'
    datasets_selection = 'GunPoint'
    model_names = ['FCN', ] #'FCN','MLP', 'InceptionTime','MLP'
    for model_name in model_names:
        CF_names = ['NUN_CF','wCF','TSEvo','SETS'] if model_name == 'MLP' else  ['NUN_CF','NG','wCF','TSEvo','SETS']
        df_threL0_sensitivity:pd.DataFrame = compute_threshold_L0_models(model_name, CF_names, datasets_selection,
                                                                         threshold,min_gap)

    # min_gap = None
    # model_name = 'FCN'
    # CF_names = 'wCF'
    # datasets_selection = 'GunPoint'
    # compute_threshold_L0_datasets(model_name, CF_names, datasets_selection,
    #                                                                   threshold, min_gap)


    # datasets_selection = 'selected_mul'
    # model_names = ['InceptionTime','FCN','MLP'] #'FCN','MLP',
    # CF_names = ['TSEvo','NUN_CF','COMTE','wCF', 'SETS'] #'NUN_CF','COMTE','wCF', 'SETS'
    # for model_name in model_names:
    #     # CF_names = ['NUN_CF','wCF','TSEvo','SETS'] if model_name == 'MLP' else  ['NUN_CF','NG','wCF','TSEvo','SETS']
    #     df_threL0_sensitivity:pd.DataFrame = compute_threshold_L0_models(model_name, CF_names, datasets_selection,
    #                                                                      threshold,min_gap)